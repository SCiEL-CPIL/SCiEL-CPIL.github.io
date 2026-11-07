import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using GP-LCB with cost normalization.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP std dev predictions  
    - f_best: float - best observed value so far
    - cost: array of shape (2000,) - evaluation cost for each candidate
    
    Returns:
    - scores: array of shape (2000,) - higher means evaluate next
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-8, None)
    
    # GP-LCB: mu - beta * sigma for minimization (lower is better)
    # We want higher scores to be better, so we negate the LCB
    beta = 2.0
    lcb = mu - beta * sigma
    
    # Higher is better, so we negate
    scores = -lcb / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
