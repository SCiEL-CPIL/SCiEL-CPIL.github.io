import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Branin 2D.
    Uses Upper Confidence Bound (UCB) for minimization.
    Score = -mu + beta * sigma. Higher score means higher priority.
    """
    # Explicit deadline guard
    start_time = time.time()
    # The computation is O(N), so it should be very fast.
    # We set a generous internal limit to satisfy the directive, 
    # but the real constraint is the 300s wall-clock for the whole evaluation loop.
    
    # UCB parameter
    # A value of 1.0 provides a balance between exploitation (low mu) 
    # and exploration (high sigma).
    beta = 1.0
    
    # Calculate UCB scores for minimization
    # We want to find the minimum, so we maximize the "lower confidence bound"
    # which corresponds to -mu + beta * sigma in the maximization of the score.
    scores = -mu + beta * sigma
    
    # Check if we are running out of time (unlikely for this simple op)
    if time.time() - start_time > 1.0:
        # Fallback to simple mean-based score if something goes wrong
        return -mu
        
    return scores
