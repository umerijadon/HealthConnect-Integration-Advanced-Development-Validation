# HealthConnect-Integration-Advanced-Development-Validation

**AnalystLab Africa Experience Lab | Data Science Track**

Predicting appointment no-shows for a fictional healthcare provider, HealthConnect Clinic, with the goal of helping the clinic act before an appointment is missed.

> **Main question:** How can HealthConnect Clinic use data and AI to reduce missed appointments and improve the patient support experience?

This repository contains my work on the Data Science track of the HealthConnect Experience Lab. Other tracks, including Project Management, Data Analytics, Machine Learning Engineering, and Generative AI, worked on different parts of the same business problem and dataset.

---

## Week 4: Problem Understanding & Initial Assessment

The first step was understanding what was actually in the dataset before building anything.

A few things stood out:

* No duplicate IDs or obvious logical issues were found. For example, `previous_no_shows` never exceeded `previous_appointments`, and `booking_lead_days` was never negative.
* After removing cancellations, the no-show and attended groups were close to balanced at roughly 51% and 49%.
* `previous_no_shows` was the clearest early signal. No-show rates increased from around 46% to 70% as previous no-shows increased.
* `booking_lead_days` was another important signal.
* Missing values were limited to three columns. The missing `reminder_channel` values turned out to be structural rather than random.
* `waiting_time_minutes` raised a possible leakage issue because I needed to confirm when that information would actually be available.

---

## Week 5: Data Preparation, Feature Engineering & Baseline Modelling

### Data preparation

| Column                  | Issue             | What I did                                                                                                                                                                      |
| ----------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `reminder_channel`      | 1,366 missing     | Confirmed that values were missing exactly when no reminder was sent, so I filled them with an explicit `"None"` category.                                                      |
| `distance_to_clinic_km` | 90 missing (1.8%) | No clear pattern was found, so I used the median and added a `distance_missing_flag`.                                                                                           |
| `waiting_time_minutes`  | Potential leakage | Removed it completely. The issue wasn't missing data. The problem was that waiting time is only known after the appointment, while the model is supposed to predict beforehand. |

### Features

I created a few features to give the model more useful information:

* `no_show_rate_history`: previous no-shows divided by previous appointments. This gives more context than simply using the raw count.
* `is_first_appointment`: identifies patients with no previous HealthConnect history.
* `short_lead_time` / `long_lead_time`: flags for appointments booked within 3 days or 30+ days in advance.
* `is_weekend_appt`: whether the appointment falls on Saturday or Sunday.
* `distance_missing_flag`: keeps track of whether the clinic distance was originally unknown.

### Train/test split

Because some patients have multiple appointments, a normal row-level split could put the same patient in both the training and test sets.

I used `GroupShuffleSplit` with `patient_id` instead, so each patient's appointments stayed entirely in either the training or test set. I also checked the result and confirmed there was zero patient overlap.

### Baseline results

| Model                                |  Accuracy | Precision |    Recall |        F1 |   ROC-AUC |
| ------------------------------------ | --------: | --------: | --------: | --------: | --------: |
| Majority-class baseline              |     0.500 |         – |         – |         – |         – |
| Rule-based (`previous_no_shows ≥ 2`) |     0.516 |     0.576 |     0.118 |     0.196 |         – |
| **Logistic Regression**              | **0.630** | **0.632** | **0.625** | **0.629** | **0.678** |
| Decision Tree (depth=5)              |     0.617 |     0.613 |     0.636 |     0.624 |     0.663 |

Logistic Regression was the best baseline and also gave me an interpretable starting point. `previous_no_shows` was the main risk-increasing factor, while `no_show_rate_history` and `reminder_sent` were among the factors associated with lower risk.

---

## Week 6: Error Analysis, Cross-Track Integration & Model Validation

Rather than just rebuilding the Week 5 model, I wanted to understand **where it was getting things wrong** and whether the findings from the Data Analytics track actually helped when introduced into the model.

### What the baseline was getting wrong

The Week 5 model had a problem with the simple lead-time flags.

**False negatives:** 178 no-shows were missed. These patients generally had few previous no-shows, but their average lead time was around 20 days. The old ≤3 / ≥30 day flags didn't really capture this middle range.

**False positives:** 176 patients were flagged even though they didn't end up missing their appointments. They had fairly ordinary histories but a long average lead time of around 39 days.

This made me rethink the binary lead-time features. The change wasn't based only on the Data Analytics findings. The error analysis showed that the original approach was simply too broad.

---

### Cross-track feature validation

The Data Analytics track shared three crosstab findings with me. Before using them, I checked each one against my cleaned dataset.

| Finding                                 | What I found                                                                                                                                                                                                                              | Used in model?                           |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| `previous_no_shows × reminder_sent`     | Held up well from 0 to 3 prior no-shows. At 2 prior no-shows, the no-reminder group had a 70.7% no-show rate compared with 58.8% for those who received a reminder. The reversal at 4+ prior no-shows came from very small samples (n≤7). | ✅ Yes                                    |
| `booking_lead_days` bands               | The clearest finding. No-show rates increased steadily from 32.8% for 0–15 days to 71.4% for 45+ days, with reasonable sample sizes across the bands.                                                                                     | ✅ Yes                                    |
| `distance_to_clinic_km × reminder_sent` | Looked reasonable at shorter distances but became noisy beyond 30km because of small samples. I tested the interaction, but it had almost no effect in the final model.                                                                   | ⚠️ Tested, but not useful enough to keep |

This was an important part of the collaboration for me. A relationship can look strong in a crosstab and still add very little once it's tested as a predictive feature.

---

### Single split vs. cross-validation

Week 5 used a single 80/20 split. For Week 6, I moved to **5-fold `GroupKFold` cross-validation**, still grouping by patient.

That changed how I viewed the improvement.

| Evaluation             | Week 5 baseline | Week 6 refined | Takeaway                          |
| ---------------------- | --------------: | -------------: | --------------------------------- |
| Single split accuracy  |          0.6335 |         0.6335 | Looks like no change              |
| Single split ROC-AUC   |           0.678 |          0.679 | Almost identical                  |
| **5-fold CV accuracy** |       **0.624** |      **0.629** | **Small, consistent improvement** |
| **5-fold CV ROC-AUC**  |       **0.680** |      **0.680** | Essentially unchanged             |

The single split made the two versions look almost identical. Cross-validation gave a better picture and showed that the refined version was consistently, although only slightly, better across the five folds.

---

### Model comparison

I also compared the refined Logistic Regression with Random Forest and Gradient Boosting using the same 5-fold grouped cross-validation setup.

| Model                             |  Accuracy | Precision | Recall |        F1 |   ROC-AUC |
| --------------------------------- | --------: | --------: | -----: | --------: | --------: |
| Week 5 baseline                   |     0.624 |     0.635 |  0.623 |     0.629 |     0.680 |
| **Logistic Regression (refined)** | **0.629** | **0.642** |  0.620 | **0.631** | **0.680** |
| Random Forest                     |     0.626 |     0.648 |  0.588 |     0.616 |     0.676 |
| Gradient Boosting                 |     0.621 |     0.632 |  0.621 |     0.626 |     0.666 |

For now, the refined Logistic Regression is the best candidate. The ensemble models didn't beat it at their current settings.

---

## Key Limitations

There are a few things I don't want the model results to hide:

* **The data is synthetic.** The patterns are useful for this project, but they would need to be checked against real clinic data before being used in practice.
* **There is no reason-for-absence data.** Knowing why patients miss appointments would make it easier to design more targeted interventions.
* **The improvement is modest.** ROC-AUC stayed around 0.680 to 0.685 after refinement. The model improved, but not by much.
* **The ensemble models were not tuned yet.** Random Forest and Gradient Boosting didn't outperform Logistic Regression at their current settings.
* **Distance × reminder wasn't useful enough in the final model.** It had some directional support in the data, but didn't provide meaningful predictive value.
* **Very high previous-no-show counts are unreliable.** There are too few patients with 4+ previous no-shows to draw strong conclusions.
* **Ethical consideration:** A no-show prediction should be used to offer additional support, not to restrict a patient's access to care.

---

## Week 7 Roadmap

* [ ] Tune Random Forest and Gradient Boosting using the same grouped cross-validation setup
* [ ] Test the specific edge cases found during error analysis
* [ ] Set a decision threshold based on the cost of a missed appointment versus an unnecessary reminder
* [ ] Revisit the day-of-week features, which still look inconsistent
* [ ] Share the final feature set and preprocessing pipeline with the ML Engineering track

---

## Tools Used

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Scikit-learn` · `Jupyter Notebook`

---

## Cross-Track Collaboration

### Week 5

I compared notes with the **Data Analytics** track on their KPI findings, including overall no-show rate and no-show rates by reminder status and previous history. This helped keep the figures reported across both tracks consistent and supported the decision to keep `reminder_channel` at its full category level rather than reducing it to a simple yes/no variable.

### Week 6

The Data Analytics track shared three findings around previous no-shows and reminders, booking lead time, and distance and reminders.

I independently checked all three against my cleaned dataset. Two translated into useful model features. The distance × reminder interaction didn't add enough predictive value to keep in the final model, so I reported that back rather than assuming that every interesting crosstab had to become a model feature.

---

*Part of the AnalystLab Africa Experience Lab: HealthConnect Clinic project. #AnalystLabAfrica*

