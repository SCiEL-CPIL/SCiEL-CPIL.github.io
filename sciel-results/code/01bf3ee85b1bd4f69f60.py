# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function for minimization.
    
    Args:
        mu: array of shape (2000,) with posterior means in objective units.
        sigma: array of shape (2000,) with posterior standard deviations in objective units.
        f_best: float, the lowest observed value so far.
    
    Returns:
        array of shape (2000,) with scores; higher means more promising to evaluate next.
    """
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0)
    
    # UCB for minimization:
    # We want to maximize the lower bound of the function value.
    # Lower bound: mu - beta * sigma
    # To make "higher score = better", we use:
    # Score = -mu + beta * sigma
    
    # Choose beta. 
    # A constant beta is used. 1.0 is a standard starting point.
    beta = 1.0
    
    scores = beta * sigma - mu
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
