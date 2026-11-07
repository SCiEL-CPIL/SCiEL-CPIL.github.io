"""Rewritten program using Cost-Weighted LCB for structural change."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global time tracking for safety margin
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    
    # Initialize time if not set
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: 330s limit, return safe result if close to limit
    # We don't have a "best result found" to return easily without state, 
    # but we can return a neutral score (e.g., uniform random or zeros) 
    # or just the standard calculation if it's fast.
    # The calculation is vectorized and fast, so this is mostly a formality 
    # unless the input size changes drastically.
    elapsed = time.time() - _start_time
    if elapsed > 320:
        # Return a valid array of zeros or random to satisfy interface
        return np.zeros_like(mu)

    # Ensure sigma is strictly positive to avoid division by zero or NaNs
    sigma = np.maximum(sigma, 1e-9)
    
    # Cost-Weighted Lower Confidence Bound (LCB)
    # We want to minimize f(x). 
    # LCB(x) = mu(x) - beta * sigma(x)
    # We want to maximize the "improvement" over f_best.
    # Approximate Improvement = f_best - LCB(x) = f_best - mu(x) + beta * sigma(x)
    
    # Adaptive Beta: 
    # Using a fixed beta is a structural choice. 
    # Beta=2.0 is a common default for BO.
    # We can scale it slightly based on the number of candidates if we wanted, 
    # but for 2000 candidates, a fixed value is stable.
    beta = 2.5
    
    # Calculate the LCB-based improvement score
    improvement_approx = f_best - mu + beta * sigma
    
    # We only want to consider points with positive expected improvement.
    # If improvement_approx <= 0, the point is unlikely to be better than f_best.
    # We clip to 0 to avoid selecting points that are clearly worse, 
    # unless all points are worse (in which case we might want to explore the least bad).
    # However, dividing by cost makes negative scores very negative (if cost is small).
    # Let's clip to a small epsilon or 0.
    scores = np.maximum(improvement_approx, 0.0)
    
    # Normalize by cost
    # cost is shape (2000,), values > 1
    scores = scores / cost
    
    # Handle any potential NaNs or Infs
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
