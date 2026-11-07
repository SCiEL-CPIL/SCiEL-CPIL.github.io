import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Upper Confidence Bound (UCB) for minimization
    # Higher score is better.
    # For minimization, we want low mu and high sigma (uncertainty).
    # UCB = -mu + beta * sigma
    # We normalize by cost to prefer cheaper evaluations.
    beta = 2.5
    ucb = -mu + beta * sigma
    score = ucb / cost
    return score
