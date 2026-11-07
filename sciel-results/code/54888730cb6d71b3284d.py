"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) acquisition function.
    
    Parameters:
    mu: array-like, GP mean predictions (2000,)
    sigma: array-like, GP standard deviation predictions (2000,)
    f_best: float, best observed function value
    cost: array-like, cost of evaluating each candidate (2000,)
    
    Returns:
    scores: array-like, acquisition scores (2000,)
    """
    # UCB exploration parameter
    # Tuned to balance exploration and exploitation for 2D Ackley
    beta = 1.5
    
    # Cost penalty exponent
    # Using 0.8 based on observed performance improvements over 1.0
    # This softens the penalty for higher-cost points, allowing more exploration
    cost_exponent = 0.8
    
    # Calculate UCB score for minimization
    # We maximize -mu + beta * sigma
    ucb_scores = -mu + beta * sigma
    
    # Apply cost-aware scaling
    # cost is in [1, 2]
    cost_factor = np.power(cost, cost_exponent)
    
    # Avoid division by zero (though cost >= 1)
    cost_factor = np.maximum(cost_factor, 1e-6)
    
    scores = ucb_scores / cost_factor
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
