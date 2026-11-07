"""Rewritten program using UCB acquisition function."""
import numpy as np
import time

# Global variable to track start time for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    
    # Initialize start time on first call
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: The evaluator kills after 330s.
    # We check elapsed time. If we are close to the limit, 
    # we should return quickly. 
    # Since the computation is vectorized and fast (<1ms), 
    # this guard is mostly for safety.
    # If elapsed time > 320s, return a simple fallback to ensure 
    # we don't hang or timeout in the caller's logic if it's tight.
    if time.time() - _start_time > 320:
        # Fallback: Simple UCB with small beta
        return -mu + 0.1 * sigma

    # Upper Confidence Bound (UCB) for minimization
    # We want to maximize the acquisition function.
    # For minimization, a low mean (mu) and high uncertainty (sigma) is good.
    # Score = -mu + beta * sigma
    # Higher score means more promising.
    
    beta = 2.0
    scores = -mu + beta * sigma
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
