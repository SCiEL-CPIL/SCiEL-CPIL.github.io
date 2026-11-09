import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Upper Confidence Bound (UCB) for minimization
    # Adaptive beta based on current best value
    # UCB = -mu + beta * sigma
    # Score = UCB / cost
    
    # Adaptive exploration parameter
    # Ensure beta is at least 1.0, but scales with f_best if it's large
    beta = max(1.0, 0.1 * f_best)
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate UCB
    ucb = -mu + beta * sigma_safe
    
    # Cost-aware: higher score for lower cost
    scores = ucb / cost
    
    # Handle any non-finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
