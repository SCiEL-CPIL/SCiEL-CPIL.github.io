import numpy as np

def acquisition(mu, sigma, f_best):
    # UCB (Upper Confidence Bound) for minimization
    # We want to minimize f, so we want low mu and high sigma.
    # Score = -mu + beta * sigma
    # Higher score is better.
    beta = 1.0
    return -mu + beta * sigma
