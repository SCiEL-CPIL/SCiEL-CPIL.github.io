import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0 # 5 seconds margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        # Fallback: simple EI without cost penalty to ensure fast return
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware adjustment: Cost-Effective Improvement (CEI)
    # Score = EI / cost
    # Cost is in [1, 2]
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = ei / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
