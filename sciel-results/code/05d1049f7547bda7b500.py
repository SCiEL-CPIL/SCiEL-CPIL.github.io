import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Using Upper Confidence Bound (UCB) for minimization
    # Score = -mu + beta * sigma
    # Higher score is better.
    # beta controls exploration vs exploitation.
    beta = 2.5
    return -mu + beta * sigma
