"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = 300.0  # Seconds


def acquisition(mu, sigma, f_best):
    # Batch-aware LCB with repulsion term for spatial diversity
    # LCB = mu - kappa * sigma
    # Add a repulsion penalty to avoid clustering near the same region
    
    # Check time budget
    if time.time() > _DEADLINE - 5:
        # Fallback to aggressive exploration if running out of time
        return -mu + 3.0 * sigma

    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Adaptive kappa: larger when uncertainty is high relative to best
    # Use a fixed moderate value for stability
    kappa = 1.5
    
    # Compute LCB scores
    lcb = mu - kappa * sigma_safe
    
    # Add a diversity-promoting term based on sigma itself
    # Higher sigma regions get a small bonus to encourage exploration
    # This acts as a simple repulsion from well-explored (low sigma) areas
    diversity_bonus = 0.1 * sigma_safe
    
    # Combine: higher score means more promising
    # For minimization, we want low mu and high sigma
    score = lcb + diversity_bonus
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
