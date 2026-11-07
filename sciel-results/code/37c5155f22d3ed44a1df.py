import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound for minimization
    # Score = -mu + beta * sigma
    # Higher score is better (lower mu, higher sigma)
    beta = 1.0
    return -mu + beta * sigma
