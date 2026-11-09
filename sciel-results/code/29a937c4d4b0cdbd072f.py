import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive beta
    # Calculate relative uncertainty
    mu_std = np.std(mu)
    sigma_mean = np.mean(sigma)
    
    # If mu is constant, beta should be high to explore
    if mu_std < 1e-6:
        beta = 2.0
    else:
        # Scale beta with uncertainty relative to the spread of means
        beta = 1.5 + 0.5 * (sigma_mean / (mu_std + 1e-9))
        
    # Cap beta to prevent excessive exploration
    beta = np.clip(beta, 1.0, 5.0)
    
    # LCB: mu - beta * sigma
    # We want to minimize this, so we look for low values.
    # Score should be high for good points.
    # Improvement = f_best - (mu - beta * sigma)
    lcb = mu - beta * sigma
    improvement = f_best - lcb
    
    # Only count positive improvements
    improvement = np.maximum(improvement, 0.0)
    
    # Cost-aware score
    # Avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = improvement / cost_safe
    
    # Return finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
