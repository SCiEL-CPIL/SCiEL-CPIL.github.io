"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function.
    
    Args:
        mu: array of shape (2000,) with posterior means in objective units.
        sigma: array of shape (2000,) with posterior standard deviations in objective units.
        f_best: float, the lowest observed value so far.
    
    Returns:
        array of shape (2000,) with scores; higher means more promising to evaluate next.
    """
    # Ensure sigma is positive to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # UCB parameter: balance between exploitation and exploration
    # A value around 1.0-2.0 is typical for UCB
    # We use a slightly aggressive exploration coefficient to find the minimum faster
    beta = 1.5
    
    # UCB score: mu + beta * sigma
    # Since we are minimizing, we want low mu and high uncertainty (sigma)
    # Higher UCB score means we think this point could be better (lower) than current best
    scores = mu + beta * sigma
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
