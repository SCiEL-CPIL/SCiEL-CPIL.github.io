"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB).
    
    Args:
        mu: Array of posterior means (shape 2000,).
        sigma: Array of posterior standard deviations (shape 2000,).
        f_best: Scalar, the lowest observed function value.
        cost: Array of evaluation costs for each candidate (shape 2000,).
        
    Returns:
        Array of scores (shape 2000,). Higher scores are preferred.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # UCB parameter: controls exploration vs exploitation
    # Higher beta favors exploration (high sigma)
    beta = 2.5 
    
    # Cost penalty exponent
    # alpha = 1.0 means linear penalty for cost
    # alpha = 0.5 means softer penalty
    alpha = 1.0
    
    # Calculate UCB score for minimization
    # We want to find points with low mu and high sigma.
    # Maximizing (beta * sigma - mu) achieves this.
    ucb_score = beta * sigma - mu
    
    # Apply cost-awareness
    # We divide by cost^alpha to prefer cheaper evaluations
    # Ensure cost is not zero to avoid division errors
    safe_cost = np.maximum(cost, 1e-8)
    
    scores = ucb_score / (safe_cost ** alpha)
    
    # Sanitize scores to ensure they are finite
    # If sigma is 0, ucb is just -mu. If mu is very negative, score is high.
    # If cost is very low, score is high.
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
