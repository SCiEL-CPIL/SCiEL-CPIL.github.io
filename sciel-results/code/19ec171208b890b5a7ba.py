import numpy as np

def acquisition(mu, sigma, f_best):
    # UCB for minimization
    # We want to minimize f(x).
    # Score = -mu + beta * sigma
    # Higher score is better.
    
    beta = 2.0
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-10)
    
    scores = -mu + beta * sigma_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
