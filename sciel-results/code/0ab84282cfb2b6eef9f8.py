import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle zero sigma to avoid issues, though WCI is robust to it
    # If sigma is 0, WCI = max(0, f_best - mu)
    # If sigma is large, WCI becomes negative -> 0
    
    # Exploration constant k
    # k=1.0 is a standard choice for WCI (1-sigma penalty)
    k = 1.0
    
    # Calculate Worst-Case Improvement
    # wci = (f_best - mu) - k * sigma
    # If this is negative, the point is not "safe" enough to guarantee improvement
    # We clip at 0 because we only care about potential improvement
    wci = (f_best - mu) - k * sigma
    wci = np.maximum(wci, 0.0)
    
    # Cost-aware normalization
    # Maximize improvement per unit cost
    # Cost is in [1, 2]
    cost_safe = np.maximum(cost, 1e-9)
    scores = wci / cost_safe
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    if time.time() - start_time > 290:
        return np.zeros_like(mu)
        
    return scores
