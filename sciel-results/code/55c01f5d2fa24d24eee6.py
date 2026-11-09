import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization
    # Higher score means more likely to be evaluated
    # We minimize f, so we want low mu and high sigma (uncertainty)
    beta = 1.0
    return -mu + beta * sigma
