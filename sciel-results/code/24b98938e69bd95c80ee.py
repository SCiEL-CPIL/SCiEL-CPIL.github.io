import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > 295:
        # Fallback: simple UCB
        sigma_safe = np.maximum(sigma, 1e-9)
        return (f_best - mu) + 2.0 * sigma_safe

    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration term
    # Adding a linear sigma term encourages exploration of high-uncertainty regions
    # even if the mean is currently worse than f_best.
    exploration_weight = 1.0 
    score = ei + exploration_weight * sigma_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
