"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function for minimization.
    Balances exploitation (low mu) and exploration (high sigma).
    Score is higher for points that are promising or uncertain.
    """
    # Clip sigma to avoid division by zero and ensure positivity
    sigma = np.maximum(sigma, 1e-8)
    
    # Exploration-exploitation trade-off parameter
    beta = 1.0
    
    # UCB for minimization: minimize (mu - beta * sigma)
    # We return higher scores for better candidates, so:
    # Score = -(mu - beta * sigma) = beta * sigma - mu
    scores = beta * sigma - mu
    
    # Ensure all scores are finite
    # Replace non-finite values with a very low score (low priority)
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
