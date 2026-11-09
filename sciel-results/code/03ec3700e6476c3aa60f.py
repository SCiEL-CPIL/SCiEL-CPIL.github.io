import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware UCB (Upper Confidence Bound) for minimization.
    # This approach uses a linear combination of the mean and standard deviation
    # to balance exploration and exploitation, distinct from the non-linear
    # Expected Improvement used in the seed program.
    
    # Hyperparameter controlling the degree of exploration
    beta = 1.0
    
    # For minimization, the UCB value is mu - beta * sigma.
    # We want to maximize the acquisition score, so we invert the UCB logic
    # relative to the current best f_best.
    # Score is high when (f_best - (mu - beta * sigma)) is large.
    # This simplifies to f_best - mu + beta * sigma.
    
    ucb_score = f_best - mu + beta * sigma
    
    # Cost-aware scaling: prioritize cheaper evaluations
    # cost is guaranteed to be >= 1.0 by the problem description
    score = ucb_score / cost
    
    # Ensure the output is finite (mu, sigma, f_best, cost are finite)
    return score
