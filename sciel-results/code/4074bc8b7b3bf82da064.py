"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we are extremely close to the limit, return a safe fallback
    # In practice, this loop runs in < 1ms, so this is just a safety net.
    elapsed = time.time() - _start_time
    if elapsed > 325:
        # Fallback: return -mu (exploit current best) to ensure finite output
        return -np.asarray(mu)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Robustness: ensure sigma is strictly positive to avoid division by zero or negative variance issues
    # This refinement is taken from Inspiration 1
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Compute EI
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure all values are finite to satisfy the contract
    # This refinement is taken from Inspiration 1
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return ei
# EVOLVE-BLOCK-END
