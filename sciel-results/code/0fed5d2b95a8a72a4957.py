import numpy as np
from scipy.stats import norm
import time

# Global variable to track start time for deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 285.0  # Safety margin before 300s

def acquisition(mu, sigma, f_best):
    # Deadline guard: If we are very close to the time limit, 
    # return a score that strongly favors exploitation (low mu) 
    # to avoid wasting time on exploration or complex calculations.
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback to pure exploitation: lower mu is better
        # Invert mu so higher score = lower mu
        return -mu
    
    # Ensure sigma is safe for division
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Standard Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(ei), ei, 0.0)
    
    return score
