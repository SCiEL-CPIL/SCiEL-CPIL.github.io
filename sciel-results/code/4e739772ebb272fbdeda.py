"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 5.0


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Probability of Improvement (PoI).
    mu, sigma: arrays of shape (2000,)
    f_best: scalar, lowest observed value
    Returns: array of 2000 finite scores, higher = more promising
    """
    # Deadline guard
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        # Return zeros if we're about to timeout
        return np.zeros_like(mu)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best) = Phi(z)
    poi = norm.cdf(z)
    
    # Ensure finite output
    poi = np.where(np.isfinite(poi), poi, 0.0)
    
    return poi
# EVOLVE-BLOCK-END
