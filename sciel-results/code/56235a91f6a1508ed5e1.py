import numpy as np
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
    
    # Worst-Case Improvement (WCI): deterministic exploration-exploitation tradeoff
    # WCI = max(0, f_best - mu - k * sigma)
    # This penalizes high uncertainty regions (large sigma) to avoid wasting budget
    # while still exploiting regions where mu is low.
    
    # Use a fixed exploration constant k. 
    # k=1.0 balances exploration and exploitation.
    # Smaller k favors exploitation (greedy), larger k favors exploration.
    k = 1.0
    
    wci = f_best - mu - k * sigma_safe
    
    # Clip to zero for points that are not expected to improve
    wci = np.maximum(wci, 0.0)
    
    # Cost-aware score: higher WCI per unit cost is better
    scores = wci / cost_safe
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    if time.time() - start_time > 290:
        return np.zeros_like(mu)
        
    return scores
