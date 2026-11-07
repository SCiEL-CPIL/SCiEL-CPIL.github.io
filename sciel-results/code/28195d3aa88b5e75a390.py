"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d).
Uses Cost-Adjusted Lower Confidence Bound (LCB) strategy, which is structurally
different from Expected Improvement. LCB explicitly encourages exploration of
high-uncertainty regions by subtracting a multiple of the standard deviation
from the mean, then inverting the sign to maximize the score.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline management
_START_TIME = time.monotonic()
_DEADLINE_SECONDS = 290.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted LCB.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far (unused in LCB, kept for interface)
    cost : array of shape (2000,)
        Evaluation cost for each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Check deadline
    elapsed = time.monotonic() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        # Return a safe, finite, valid score to avoid invalidation
        return np.zeros_like(mu, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=-1e6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    
    # Ensure cost is within expected range [1, 2]
    cost = np.clip(cost, 1.0, 2.0)
    sigma = np.maximum(sigma, 1e-12)
    
    # LCB parameter: controls exploration
    # Higher beta means more exploration
    beta = 2.0
    
    # LCB for minimization: mu - beta * sigma
    # We want to maximize the score, so we negate the LCB
    # Score proportional to -(mu - beta * sigma) = beta * sigma - mu
    lcb_score = beta * sigma - mu
    
    # Cost adjustment: divide by cost to favor cheaper evaluations
    score = lcb_score / cost
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
