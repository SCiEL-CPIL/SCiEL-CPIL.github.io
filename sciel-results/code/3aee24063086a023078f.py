import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound acquisition function
    # Minimizing: we want low mean, high variance.
    # Score is higher for better candidates.
    beta = 2.0
    return -mu + beta * sigma
