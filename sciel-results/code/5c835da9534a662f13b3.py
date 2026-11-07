import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Check for timeout (unlikely for 2000 points, but required by directive)
    if time.time() - start_time > 290:
        # Fallback to a simple heuristic if we are somehow slow
        # This branch is effectively dead code for 2000 points
        return np.zeros_like(mu)
        
    # Thompson Sampling
    # Sample from the posterior distribution
    samples = np.random.normal(mu, sigma)
    
    # Cost-aware score: we want to minimize samples / cost
    # So we maximize -samples / cost
    score = -samples / cost
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
