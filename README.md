# Predicting adult salary

This repository contains a Jupyter Notebook for training and analyzing machine learning models on the Adult dataset. The purpose of this script is to predict whether an individual's income exceeds $50K/year based on various demographic features.

## Dataset
The dataset used in this analysis is the Adult dataset, sourced from the UCI Machine Learning Repository. It contains demographic information such as age, education, occupation, etc., and the target variable is the income level categorized as either >50K or <=50K.

## Prerequisites
Before running the script, make sure you have the following packages installed:
- ucimlrepo
- seaborn
- Numpy
- Pandas
- Sklearn
- Matplotlib.pyplot
- python 3.10.13

You can install them using pip:

```bash
pip install -U ucimlrepo 
pip install -U seaborn 
pip install -U numpy 
pip install -U pandas 
pip install -U scikit-learn 
pip install -U matplotlib
```

## Usage
1. Clone this repository to your local machine.
2. Open the provided Python script in your preferred Python environment (e.g., Jupyter Notebook).
3. Run the script cell by cell to execute each step of the analysis and model training.
4. The script includes data preprocessing, exploratory data analysis, model training, evaluation, and selection steps.

## Summary of Steps
1. **Data Loading**: Fetch the Adult dataset from the UCI Machine Learning Repository.
2. **Data Cleaning**: Handle missing values, duplicate entries, and format the target variable.
3. **Exploratory Data Analysis**: Visualize data distributions, explore relationships between features and the target variable.
4. **Feature Engineering**: Encode categorical variables using one-hot encoding.
5. **Model Training**: Train multiple machine learning models including Logistic Regression, Decision Tree, Random Forest, Gaussian Naive Bayes, and Support Vector Machine (SVM).
6. **Model Evaluation**: Evaluate each model's performance using metrics like precision, recall, and F1-score.
7. **Model Selection**: Choose the best-performing model based on the evaluation metrics.

## Results
- The Logistic Regression model achieved the highest F1-score, making it the recommended model for predicting income levels.
- Detailed results including model parameters, precision, recall, and F1-score are provided in the notebook.

## Conclusion
The Logistic Regression model trained on the Adult dataset demonstrates good performance in predicting income levels. Further optimization and fine-tuning of the model parameters may lead to even better results.

