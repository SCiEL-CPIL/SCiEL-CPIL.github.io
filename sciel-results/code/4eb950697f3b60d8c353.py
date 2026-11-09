import numpy as np
from scipy.stats import norm
import time

_START_TIME = None

def _get_elapsed_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def acquisition(mu, sigma, f_best, cost):
    # Check time limit
    elapsed = _get_elapsed_time()
    # If we are close to the 330s limit, use a very fast fallback
    if elapsed > 320.0:
        # Fallback: simple cost-aware regret
        # Ensure finite output
        scores = (f_best - mu) / (cost + 1e-9)
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    # Standard Cost-Aware Expected Improvement
    # Handle sigma near zero to avoid division by zero
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * CDF(z) + sigma * PDF(z)
    # Using norm.cdf and norm.pdf
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Cost-aware
    scores = ei / (cost + 1e-9)
    
    # Ensure finite values
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
