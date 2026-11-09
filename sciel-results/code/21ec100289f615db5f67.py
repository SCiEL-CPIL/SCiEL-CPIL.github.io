import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Using Upper Confidence Bound (UCB) for minimization
    # Maximizing -mu + beta * sigma
    beta = 2.0
    return -mu + beta * sigma
