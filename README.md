# Web Scraping for Indeed.com and Predicting Salaries

The **goal** if this project is to predict the range of salary of job postings based on factors such as location, 
job title, job level, industry sector.

### Procedure

1. Search on indeed.com for data scientist, data analyst, data engineer and data science jobs in different UK cities and scrape
job title, company and salary for each job posting, then add the search location to the dataframe. It's better not to scrape 
location but just append it from search location because some posting might specify the neighbourood and not the city itself.

2. Clean the salary column and only consider job posting with an explicit salary a year (mean if range is explicited). EDA 
shows that median salary is 37500£ a year so i split salaries between high and low considering the median. Baseline accuracy: 
0.5173375846950976.

3. In order to model using only location as features, I dummify locations and drop liverpool because it has the lowest mean
salary. I then use LogisticRegression with GridSearch (CV score 0.5739779903475143) and DecisionTreeClassifier with GridSearch
(CV score 0.5739817123857025) + Boosting (CV score 0.5704966439622338). By looking at the coefficients we notice that higher 
salaries are associated with cities like Cambridge and London

4. I categorise company and job titles by keywords, dummify the variables and join the new df with the dummified cities. EDA 
shows that many job postings include "analyst" in the title and the majority of these jobs are posted by recruitment companies,
which will be useless for modelling later

5. I convert the df to a sparse matrix to make calculations quicker. The LogisticRegression with GridSearch (CV score 
0.6940484609372093) indicate that keywords like "Director" and "Banking" are associated to higher salaries, whereas "Intern",
"Graduate" and "Junior" are indicator of a low salary. I also tried RandomForestClassifier with GridSearch and got a better 
CV score 0.7040111164873885. AdaBoost does not improve the score.

6. To address the problem of misclassifying a lower paying job rather than telling a client incorrectly that they would get
a high salary job, I need to minimise false positives by improving my precision score and threshold. Incresing the threshold
to 60%, thus classifying anything with less than 60% probability as low income, I make my model less accurate but with a 
higher precision. This trade-off is shown by the roc curve. 
