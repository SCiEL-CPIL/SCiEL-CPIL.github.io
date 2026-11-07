import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound for minimization
    # We maximize mu - beta * sigma
    beta = 2.0
    sigma_safe = np.maximum(sigma, 1e-9)
    scores = mu - beta * sigma_safe
    return scores
