import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety for division
    cost_safe = np.maximum(cost, 1e-9)
    
    # Worst-Case Improvement (WCI) for minimization
    # We define the worst-case value at x as mu + k * sigma.
    # The improvement is how much better f_best is than this worst-case estimate.
    # If mu + k*sigma > f_best, the worst case is worse than current best, so improvement is 0.
    # If mu + k*sigma < f_best, there is a robust chance of improvement.
    
    # Exploration constant k. 
    # A value around 1.0 to 2.0 is typical. 
    # Higher k means more conservative (exploitative), lower k means more exploratory.
    # Given the cost constraint, we want to be efficient.
    k = 1.5
    
    # Calculate WCI
    worst_case_est = mu + k * sigma
    improvement = f_best - worst_case_est
    
    # Clip to zero for non-improving regions
    wci = np.maximum(improvement, 0.0)
    
    # Cost-aware score: Divide by cost
    # Higher score = Better improvement per unit cost
    scores = wci / cost_safe
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    if time.time() - start_time > 290:
        return np.zeros_like(mu)
        
    return scores
