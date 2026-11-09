import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Cost-aware UCB
    # For minimization, we want to maximize (f_best - mu) + beta * sigma
    # Divided by cost
    beta = 5.0
    scores = (f_best - mu + beta * sigma_safe) / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
