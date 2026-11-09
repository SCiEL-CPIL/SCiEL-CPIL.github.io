import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best):
    # Deadline guard: if we're close to the time limit, return a simple score quickly
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        return -mu  # Fallback to exploitation
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10.0, 10.0)
    
    # Expected Improvement (EI) for minimization
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
