import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: handle any non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-9)
    
    # Posterior Mean Minimization (PMM): directly minimize the predictive mean
    # Score = -mu, so higher score means lower predicted value
    # This is a purely exploitative strategy that ignores uncertainty
    
    score = -mu
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
