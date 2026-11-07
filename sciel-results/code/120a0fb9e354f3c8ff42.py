"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for minimization.
    Uses Upper Confidence Bound (UCB) strategy with a linear cost penalty.
    
    Args:
        mu: GP mean predictions (shape 2000,)
        sigma: GP standard deviation predictions (shape 2000,)
        f_best: Current best observed function value
        cost: Cost of evaluating each candidate (shape 2000,)
        
    Returns:
        Scores (shape 2000,), higher is better.
    """
    # Parameters
    kappa = 2.0  # Exploration-exploitation balance
    lambda_cost = 0.5  # Cost penalty weight
    
    # Ensure sigma is non-negative and stable
    sigma = np.maximum(sigma, 1e-6)
    
    # UCB for minimization: -mu + kappa * sigma
    # Minus cost penalty: - lambda_cost * cost
    scores = -mu + kappa * sigma - lambda_cost * cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
