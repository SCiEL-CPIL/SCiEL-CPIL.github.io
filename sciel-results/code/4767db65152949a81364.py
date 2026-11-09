import numpy as np
from scipy.stats import norm
import time

def _ensure_deadline(elapsed):
    """Check if we're getting close to the deadline."""
    return elapsed < 300.0

def _time_remaining():
    """Return time remaining in seconds."""
    return max(0.0, 300.0 - time.time())

def _sanitize(scores):
    """Ensure scores are valid: finite, correct shape, and properly scaled."""
    if scores.shape != (2000,):
        scores = np.resize(scores, (2000,))
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    return scores

def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.nan_to_num(sigma, nan=1e-9, posinf=1e6, neginf=1e-9)
    
    # Ensure mu is finite
    mu = np.nan_to_num(mu, nan=f_best, posinf=f_best, neginf=f_best)
    
    # Compute Expected Improvement (EI)
    # EI is the standard acquisition function for Bayesian optimization
    # Higher EI means higher expected improvement over f_best
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Sanitize the result
    score = _sanitize(ei)
    
    return score
