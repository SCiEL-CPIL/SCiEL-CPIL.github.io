import numpy as np
from scipy.stats import norm
import time

# Global state for time tracking
_bo_start_time = None

def acquisition(mu, sigma, f_best):
    global _bo_start_time
    
    # Initialize start time on first call
    if _bo_start_time is None:
        _bo_start_time = time.time()
        
    elapsed = time.time() - _bo_start_time
    
    # Safety margin: if we are too close to the 330s limit,
    # return a simple greedy score to minimize overhead.
    # The limit is 330s, so 325s is a safe threshold.
    if elapsed > 325.0:
        # Greedy: pick the point with the lowest mean
        return -mu

    # Standard Expected Improvement for minimization
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure scores are finite
    # If there are NaNs or Infs, replace them
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return ei
