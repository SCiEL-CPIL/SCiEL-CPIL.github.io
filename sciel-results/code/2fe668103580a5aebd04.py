# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function for minimization.
    
    Args:
        mu: array of shape (2000,) with posterior means in objective units.
        sigma: array of shape (2000,) with posterior standard deviations in objective units.
        f_best: float, the lowest observed value so far. (Not used directly in UCB, but mu is conditioned on it).
    
    Returns:
        array of shape (2000,) with scores; higher means more promising to evaluate next.
    """
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0)
    
    # Exploration parameter
    # A value of 2.5 is a common starting point for 1D problems
    beta = 2.5
    
    # For minimization, we want to maximize: -mu + beta * sigma
    # Low mu is good (score increases as -mu increases)
    # High sigma is good (score increases as sigma increases)
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
