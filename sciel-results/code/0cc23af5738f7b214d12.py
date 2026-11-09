import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound for minimization
    # Maximize: -mu + beta * sigma
    beta = 1.0
    scores = -mu + beta * sigma
    # Ensure finite values (sigma is std dev, so non-negative, mu is finite)
    # If sigma is 0, score is -mu.
    return scores
