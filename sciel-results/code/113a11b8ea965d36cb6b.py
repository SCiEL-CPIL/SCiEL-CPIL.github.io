import numpy as np
import time
from scipy.stats import norm

# Global variable to track start time for the deadline guard
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB) 
    for minimization, divided by cost.
    
    Parameters:
    mu: Array of GP mean predictions (shape 2000,)
    sigma: Array of GP standard deviations (shape 2000,)
    f_best: Best observed function value (scalar)
    cost: Array of costs for each candidate (shape 2000,)
    
    Returns:
    Array of acquisition scores (shape 2000,)
    """
    global _start_time
    
    # Initialize start time on first call
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: if we are too close to the 330s limit, 
    # return a simple heuristic to ensure we finish and don't crash.
    # We use 290s as a safety margin.
    if time.time() - _start_time > 290:
        # Fallback: Prefer cheap points
        return 1.0 / (cost + 1e-9)
        
    # UCB for minimization:
    # We want to find points with low mu and high sigma (uncertainty).
    # UCB value for minimization: mu - beta * sigma
    # We want to MAXIMIZE the acquisition score.
    # So we maximize: -(mu - beta * sigma) / cost = (beta * sigma - mu) / cost
    
    # Beta controls the exploration/exploitation trade-off.
    # For 6D problems, beta=2.0 to 3.0 is common.
    beta = 2.0
    
    # Calculate score
    # Ensure cost is positive and not zero
    safe_cost = cost + 1e-9
    
    score = (beta * sigma - mu) / safe_cost
    
    # Handle any numerical issues
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Clip to avoid extreme values that might cause issues downstream
    score = np.clip(score, -1e6, 1e6)
    
    return score
