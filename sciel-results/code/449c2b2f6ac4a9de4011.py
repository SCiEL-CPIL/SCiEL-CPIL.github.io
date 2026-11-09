"""Rewritten acquisition for Bayesian optimization of Hartmann 3D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 300.0
_MARGIN = 10.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Upper Confidence Bound (UCB) for minimization.
    Returns scores where higher is better.
    Score = -mu + kappa * sigma
    """
    # Check deadline
    if time.time() > _DEADLINE - _MARGIN:
        # Return a safe fallback, e.g., pure exploitation
        return -mu

    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB parameter
    # Fixed constant to balance exploration and exploitation.
    kappa = 1.0
    
    # Higher score means more promising (lower mu, higher sigma)
    scores = -mu + kappa * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
