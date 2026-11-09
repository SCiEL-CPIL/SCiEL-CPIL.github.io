import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # UCB acquisition function for minimization
    # We want to maximize: beta * sigma - mu
    # Adjusted for cost: (beta * sigma - mu) / cost
    
    beta = 2.0
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0)
    
    # Calculate UCB score
    ucb_scores = beta * sigma - mu
    
    # Normalize by cost
    # cost is expected to be > 0
    scores = ucb_scores / cost
    
    # Ensure finite scores
    # If cost is 0 (shouldn't happen), replace with 1
    cost_safe = np.where(cost > 1e-9, cost, 1.0)
    scores = ucb_scores / cost_safe
    
    # Replace any non-finite values with a large negative number
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    return scores
