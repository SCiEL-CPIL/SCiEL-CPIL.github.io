"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Upper Confidence Bound (UCB) for minimization (Lower Confidence Bound value)
    # We want to minimize f(x). The lower bound on the minimum is mu - beta * sigma.
    # We want to maximize the potential to find a value lower than f_best.
    # A common score is (beta * sigma - mu) / cost.
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Beta controls exploration. 1.0 is a standard starting point.
    # Given the small budget, slightly higher exploration might be beneficial.
    beta = 1.5
    
    # Calculate LCB score
    # Higher score is better.
    # If mu is low and sigma is high, score is high.
    scores = (beta * sigma - mu) / cost
    
    # Handle potential numerical issues
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
