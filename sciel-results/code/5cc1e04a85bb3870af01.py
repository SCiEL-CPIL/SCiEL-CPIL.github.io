import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # UCB approach
    beta = 1.0
    
    # Ensure sigma is non-negative and handle zeros
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # UCB score for minimization: -mu + beta * sigma
    ucb_scores = -mu + beta * sigma_safe
    
    # Cost-aware: divide by cost
    # cost is >= 1, so no division by zero issues
    scores = ucb_scores / cost
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
