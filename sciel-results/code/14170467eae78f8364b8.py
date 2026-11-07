"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) acquisition function.
    
    Args:
        mu: Array of mean predictions (2000,)
        sigma: Array of standard deviation predictions (2000,)
        f_best: Scalar, the best observed value so far.
        cost: Array of costs for each candidate (2000,)
        
    Returns:
        Array of acquisition scores (2000,)
    """
    # Hyperparameter for exploration
    beta = 4.0
    
    # Ensure sigma is not zero to avoid division issues if used in other contexts, 
    # though here it's additive. But good practice.
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB for minimization: we want low mu, high sigma.
    # Value = -mu + beta * sigma
    # Cost-aware: divide by cost to penalize expensive points.
    # cost is always >= 1.
    
    scores = (-mu + beta * sigma_safe) / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
