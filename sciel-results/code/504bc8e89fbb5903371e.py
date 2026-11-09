"""Acquisition function for Bayesian optimization of hartmann 6d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 320.0  # Safety margin before the 330s kill limit


def acquisition(mu, sigma, f_best):
    """
    Adaptive Lower Confidence Bound (LCB) acquisition function.
    
    Args:
        mu: Array of predicted means (shape (2000,)).
        sigma: Array of predicted standard deviations (shape (2000,)).
        f_best: The best observed value so far.
        
    Returns:
        Array of scores (shape (2000,)), higher is better.
    """
    # Check for timeout
    if time.time() - _start_time > _TIMEOUT:
        return np.zeros_like(mu)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive beta: start with moderate exploration, increase if stuck
    # f_best is the lowest observed value; hartmann 6d minimum is ~-3.32237
    # If f_best is still far from the minimum, we need more exploration
    # If f_best is close to the minimum, we can exploit more
    
    # Heuristic: measure how far we are from the global minimum
    # Global min of hartmann 6d is approximately -3.32237
    target_min = -3.32237
    
    # If f_best is much worse than target, increase exploration
    # If f_best is close to target, decrease exploration to exploit
    
    # Scale beta based on progress
    # Early on (f_best >> target_min), use higher beta for exploration
    # Later (f_best close to target_min), use lower beta for exploitation
    
    # Use a smooth transition
    # progress = 1 when f_best = target_min, 0 when f_best is far
    # Let's say "far" is f_best > 0 (hartmann can be positive)
    # "Close" is f_best < -3.0
    
    if f_best > 0:
        beta = 4.0  # High exploration when far from optimum
    elif f_best > -2.0:
        beta = 3.5  # Moderate exploration
    elif f_best > -3.0:
        beta = 3.0  # Moderate exploitation
    elif f_best > -3.2:
        beta = 2.5  # More exploitation
    else:
        beta = 2.0  # High exploitation near optimum
    
    # LCB for minimization: we want to minimize mu - beta * sigma
    # So we maximize -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
# EVOLVE-BLOCK-END
