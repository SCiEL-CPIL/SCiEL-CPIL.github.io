import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization
    # We want to maximize the score, so we want low mu and high sigma.
    # Score = -mu + beta * sigma
    beta = 1.0
    return -mu + beta * sigma
