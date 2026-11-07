"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB).
    
    This structural change replaces the previous hybrid mixture of EI, UCB, and PI
    with a single, clean UCB score. UCB is known to perform well on multimodal
    functions like Branin by explicitly balancing exploitation (-mu) and
    exploration (beta * sigma).
    
    Args:
        mu: (2000,) array of posterior means.
        sigma: (2000,) array of posterior standard deviations.
        f_best: float, the best function value observed so far.
        
    Returns:
        (2000,) array of scores. Higher scores indicate more promising candidates.
    """
    # Convert to numpy arrays to ensure compatibility
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Exploration parameter. 
    # Beta=3.0 is chosen to strongly encourage exploration in the early stages
    # and to escape local minima in the Branin landscape.
    beta = 3.0
    
    # Compute UCB scores: -mu + beta * sigma
    # We negate mu because we are minimizing the objective, so lower mu is better.
    # Higher sigma is better for exploration.
    scores = -mu + beta * sigma
    
    # Handle any potential numerical issues (e.g., inf or nan)
    # If a score is not finite, assign it a very low value so it is never selected.
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
