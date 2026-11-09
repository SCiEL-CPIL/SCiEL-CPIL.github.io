"""Rewritten program for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 10.0 # seconds

def acquisition(mu, sigma, f_best):
    """
    Acquisition function based on Lower Confidence Bound (LCB).
    
    Args:
        mu: Mean of the GP posterior at candidates (shape 2000,).
        sigma: Std dev of the GP posterior at candidates (shape 2000,).
        f_best: Current best function value (minimum).
        
    Returns:
        scores: Array of shape 2000, higher is better.
    """
    # Deadline guard
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        # If we are too close to the deadline, return a safe fallback.
        # In this case, we can just return EI or a simple heuristic.
        # Given the speed, this is unlikely to trigger, but required by directive.
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # LCB: We want to minimize f(x).
    # The lower bound of the function value is mu - beta * sigma.
    # We want to maximize the score, so we return - (mu - beta * sigma) = -mu + beta * sigma.
    
    # Choose beta. 
    # A beta of 1.0-2.0 is typical. 
    # Since we don't know the step, we can't easily adapt beta.
    # However, we can use the current f_best and the global minimum knowledge?
    # No, we only have f_best.
    
    # Let's try a slightly more adaptive beta based on the uncertainty.
    # If sigma is very small everywhere, we are in exploitation.
    # If sigma is large, we are in exploration.
    
    # For ST1, the global min is -39.16. 
    # If f_best is much higher (e.g., -10), we need exploration.
    # If f_best is close to -39, we need exploitation.
    
    # A simple heuristic: 
    # beta = 2.0 is a good start for exploration-heavy LCB.
    beta = 2.0
    
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
