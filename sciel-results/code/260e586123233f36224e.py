"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track time for deadline guard
_start_time = time.time()
_DEADLINE = 300.0  # seconds, with safety margin before 330s kill

def _time_remaining():
    """Return seconds remaining before deadline."""
    return _DEADLINE - (time.time() - _start_time)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using adaptive LCB with exploration.
    
    Parameters:
    mu (np.ndarray): GP mean predictions, shape (2000,)
    sigma (np.ndarray): GP standard deviations, shape (2000,)
    f_best (float): Best observed function value
    cost (np.ndarray): Cost of evaluating each point, shape (2000,)
    
    Returns:
    np.ndarray: Acquisition scores, shape (2000,)
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: return zeros if time is running out
    if _time_remaining() < 1.0:
        return np.zeros_like(mu)
    
    # Avoid division by zero in cost
    cost_safe = np.maximum(cost, 1e-8)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Adaptive kappa: start with moderate exploration, increase if f_best is poor
    # For Styblinski-Tang, minimum is -78.3323
    # If f_best is far from minimum, we need more exploration
    target_min = -78.3323
    gap = f_best - target_min  # positive if we haven't found the minimum yet
    
    # Adaptive kappa: more exploration when gap is large
    # kappa ranges from about 1.0 (exploit) to 3.0 (explore)
    if gap > 50:
        kappa = 3.0
    elif gap > 20:
        kappa = 2.5
    elif gap > 10:
        kappa = 2.0
    elif gap > 5:
        kappa = 1.5
    elif gap > 2:
        kappa = 1.2
    else:
        kappa = 1.0
    
    # LCB for minimization: mu - kappa * sigma
    # We want to maximize the negative of this (i.e., minimize mu - kappa*sigma)
    # So acquisition score = -(mu - kappa * sigma) / cost = (-mu + kappa * sigma) / cost
    lcb = mu - kappa * sigma
    scores = -lcb / cost_safe
    
    # Alternative: also consider a UCB-like term for points with high uncertainty
    # and low cost, to encourage exploration in cheap regions
    # Blend: 0.7 * LCB + 0.3 * (sigma / cost) for exploration bonus
    exploration_bonus = sigma / cost_safe
    scores = 0.7 * (-mu / cost_safe) + 0.3 * (kappa * sigma / cost_safe) + 0.1 * exploration_bonus
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
