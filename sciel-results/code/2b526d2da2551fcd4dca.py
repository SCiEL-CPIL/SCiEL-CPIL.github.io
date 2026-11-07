"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _call_count
    _call_count += 1
    
    # Deadline guard: if we are too close to the limit, return a simple score
    # to ensure we don't timeout.
    if time.time() - _start_time > 325:
        # Fallback to a simple greedy score
        return -mu 
    
    # t is the number of optimization steps taken (excluding initial 5)
    # _call_count is 1 for the first optimization step
    t = _call_count
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # We want to maximize EI.
    
    # Handle case where sigma is very small or zero to avoid division by zero
    # If sigma is 0, EI is max(0, f_best - mu).
    # For numerical stability, add a small epsilon to sigma.
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, -np.inf)
    
    return ei
# EVOLVE-BLOCK-END
