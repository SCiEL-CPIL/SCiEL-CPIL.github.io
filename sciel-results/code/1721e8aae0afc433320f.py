"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # safety margin before 330s kill

def _check_deadline():
    """Return True if we should stop working (deadline approaching)."""
    return (time.time() - _start_time) > _DEADLINE

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement acquisition function for Bayesian optimization.
    
    mu, sigma: shape (2000,), posterior mean and std in objective units.
    f_best: lowest observed value so far.
    Returns: 2000 finite scores, higher = more promising.
    """
    n = len(mu)
    
    # Deadline guard: if we're close to the limit, return a simple EI score
    if _check_deadline():
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compute standard Expected Improvement
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure all scores are finite
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return ei
# EVOLVE-BLOCK-END
