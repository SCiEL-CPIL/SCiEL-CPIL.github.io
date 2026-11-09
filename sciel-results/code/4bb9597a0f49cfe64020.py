"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function for minimization.
    
    Parameters:
    mu: array-like, shape (n,), predicted mean of the GP
    sigma: array-like, shape (n,), predicted standard deviation of the GP
    f_best: float, the best (lowest) observed value
    cost: array-like, shape (n,), cost of evaluating each candidate
    
    Returns:
    scores: array-like, shape (n,), acquisition scores (higher is better)
    """
    # Convert inputs to numpy arrays for vectorized operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is positive to avoid division errors
    # Cost is defined as 1 + mean(x), so it is always >= 1
    cost = np.maximum(cost, 1e-8)
    
    # LCB for minimization: mu - beta * sigma
    # We want to MINIMIZE this value.
    # Therefore, we MAXIMIZE: beta * sigma - mu
    
    # Beta controls exploration. In 6D, higher beta encourages more exploration.
    # 3.0 is a standard choice for moderate exploration.
    beta = 3.0
    
    # Calculate the un-normalized score
    # Higher score means more promising (lower expected value relative to uncertainty)
    scores = beta * sigma - mu
    
    # Normalize by cost to penalize expensive evaluations
    scores = scores / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
