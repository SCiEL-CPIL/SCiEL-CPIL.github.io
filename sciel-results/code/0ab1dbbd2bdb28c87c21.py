"""Rewritten program using Cost-Aware Upper Confidence Bound (UCB) for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function based on Upper Confidence Bound (UCB).
    For minimization, we maximize the Lower Confidence Bound (LCB): -mu + beta * sigma.
    This approach balances exploitation (low mu) and exploration (high sigma).
    
    Args:
        mu: Array of GP means, shape (2000,)
        sigma: Array of GP standard deviations, shape (2000,)
        f_best: Scalar, lowest observed value.
        cost: Array of costs for each candidate, shape (2000,)
        
    Returns:
        Array of scores, shape (2000,), higher is better.
    """
    epsilon = 1e-9
    
    # Exploration parameter beta
    # Chosen to encourage exploration of uncertain regions to escape local minima
    beta = 2.0
    
    # LCB score for minimization: Maximize (-mu + beta * sigma)
    # -mu favors low predicted values (exploitation)
    # beta * sigma favors high uncertainty (exploration)
    lcb_scores = -mu + beta * sigma
    
    # Cost-aware score: Maximize LCB per unit cost
    scores = lcb_scores / (cost + epsilon)
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
