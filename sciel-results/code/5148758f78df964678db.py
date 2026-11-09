import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization
    # We maximize the score, which corresponds to minimizing the function.
    # Score = -mu + beta * sigma
    # Higher score means lower mu (good) or higher sigma (uncertain).
    beta = 2.5
    return -mu + beta * sigma
