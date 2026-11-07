"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None
_DEADLINE = 300.0  # seconds

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 6d.
    Uses Expected Improvement (EI) for minimization with a safety deadline guard.
    
    Parameters:
    mu: array of shape (2000,) - posterior mean
    sigma: array of shape (2000,) - posterior standard deviation
    f_best: float - best observed value (minimum)
    
    Returns:
    scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    
    # If we're getting close to the deadline, return a simple fallback
    if elapsed > _DEADLINE - 10:
        # Return sigma-based scores as a safe fallback
        sigma_safe = np.maximum(sigma, 1e-9)
        scores = sigma_safe
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Use EI as the acquisition score
    scores = ei
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
