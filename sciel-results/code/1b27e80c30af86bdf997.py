"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Upper Confidence Bound (UCB) for minimization.
    # Score = (-mu + beta * sigma) / cost
    # Higher score is better.
    
    # Ensure cost is positive to avoid division by zero or negative costs
    # Cost is defined as 1 + mean(x), so it is always >= 1.
    # But we guard against 0 just in case.
    cost_safe = np.maximum(cost, 1e-8)
    
    # Beta controls exploration. 
    # 2.0 is a standard choice for balancing exploitation and exploration.
    beta = 2.0
    
    # UCB for minimization: we want low mu, high sigma.
    # Value = -mu + beta * sigma
    ucb_value = -mu + beta * sigma
    
    # Normalize by cost
    scores = ucb_value / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
