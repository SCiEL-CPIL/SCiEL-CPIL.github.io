import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    sigma = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    return scores
