import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle zero sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate z-score for Probability of Improvement (Minimization)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate CDF (Probability of Improvement)
    poi = norm.cdf(z)
    
    # Ensure PoI is in [0, 1]
    poi = np.clip(poi, 0.0, 1.0)
    
    # Score: Probability of Improvement divided by cost
    # Higher PoI means higher chance of finding a point better than f_best.
    # Dividing by cost makes it cost-aware.
    scores = poi / cost_safe
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    if time.time() - start_time > 290:
        return np.zeros_like(mu)
        
    return scores
