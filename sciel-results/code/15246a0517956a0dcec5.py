"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for deadline guard
_bo_start_time = None
_bo_call_count = 0

def acquisition(mu, sigma, f_best):
    global _bo_start_time, _bo_call_count
    
    # Initialize start time on first call
    if _bo_start_time is None:
        _bo_start_time = time.time()
    
    _bo_call_count += 1
    
    # Deadline Guard: 330s limit, use 305s as safety margin
    if time.time() - _bo_start_time > 305:
        # If we are close to timeout, prioritize speed and exploitation
        # to ensure the loop finishes.
        return -mu
    
    # Convert to numpy arrays for safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Structural Change: Lower Confidence Bound (LCB) with adaptive beta schedule
    # LCB = mu - beta * sigma
    # This is a different acquisition family from EI: it directly trades off
    # exploitation (low mu) vs exploration (high sigma) via beta.
    # 
    # Progressive beta schedule:
    # - Early evaluations (call_count <= 10): high beta for exploration
    # - Middle evaluations (10 < call_count <= 20): moderate beta
    # - Late evaluations (call_count > 20): low beta for exploitation
    #
    # Total evaluations = 30 (5 initial + 25 BO steps), so call_count goes 1..25
    
    if _bo_call_count <= 10:
        # Exploration phase: map the landscape
        beta = 3.0
    elif _bo_call_count <= 20:
        # Transition phase: balance exploration and exploitation
        beta = 1.5
    else:
        # Exploitation phase: refine the best region
        beta = 0.5
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # LCB for minimization: we want to minimize, so we score by how low
    # the lower bound is. Higher score = more promising.
    # LCB value = mu - beta * sigma (this is the lower bound on f)
    # We want to MINIMIZE f, so we want to MAXIMIZE the score.
    # Score = -LCB = -(mu - beta * sigma) = -mu + beta * sigma
    scores = -mu + beta * sigma_safe
    
    # Ensure finite values
    scores[~np.isfinite(scores)] = -np.inf
    
    return scores
# EVOLVE-BLOCK-END
