Activity 1 Practice with a Super Learner [**]
• The goal in stacking is to ensemble strong, diverse sets of learners together.
• This involves training a second-level metalearner to "glue them" together using data.
• The algorithm that learns the optimal combination of the base learner fits is now called
the "Super Learner".
• A delicate part is how to perform the resampling to distribute and eventually reuse the
available data.
Here is an outline of the tasks involved in training and testing a Super Learner ensemble.
1. Specify a list of L base algorithms (with a specic set of hyperparameters).
2. Specify a metalearning algorithm.
3. Train each of the L base algorithms on the training set.
4. Perform k-fold cross-validation on each of these learners and collect the cross-validated
predicted values from each of the L algorithms.
5. The N cross-validated predicted values from each of the L algorithms can be combined
to form a new N x L matrix. This matrix, along wtih the original response vector, is
called the "level-one" data.
6. Train the metalearning algorithm on the level-one data.
7. The "ensemble model" consists of the L base learning models and the metalearning
model, which can then be used to generate predictions on a test set.
A useful practical resource is found in: h20. If in doubt, read the original paper, or a simpler
one. As usual, you can choose between R and python.
1. Create Super Learners for the spam data and return the pair (best model found, honest
estimation of its performance)
2. The involved MLAs (for both the level0 and level1 learners) are up to you
3. Do not try “all-with-all” models, think before combining and come up with promising
combinations beforehand
4. Decide whether you are going to incorporate a regularization mechanism or not.
5. Design an experimental methodology to test everything (resampling protocol included)
and how to choose the best model.
6. Decide or learn what is the correct way to interpret the results.


Useful links:
https://docs.h2o.ai/h2o/latest-stable/h2o-docs/starting-h2o.html

https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/stacked-ensembles.html
