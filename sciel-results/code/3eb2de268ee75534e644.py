"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    UCB-based acquisition function for Bayesian optimization of Hartmann 6d.
    Uses Upper Confidence Bound to balance exploration and exploitation.
    """
    # Deadline guard
    t_start = time.time()
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB: mu - beta * sigma (minimization, so we want low mu and high sigma)
    # beta controls exploration-exploitation tradeoff
    # For Hartmann 6d with 30 evaluations, use moderate exploration
    beta = 2.0
    
    scores = mu - beta * sigma_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety check
    if time.time() - t_start > 250:
        return scores
        
    return scores
# EVOLVE-BLOCK-END
