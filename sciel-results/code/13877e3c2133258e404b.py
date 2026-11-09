"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global timer to track elapsed time for the deadline guard
_start_time = time.time()
TIME_LIMIT = 300.0
SAFETY_MARGIN = 10.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Lower Confidence Bound (LCB) for minimization.
    Score = -mu + beta * sigma
    """
    # Check time
    elapsed = time.time() - _start_time
    if elapsed > TIME_LIMIT - SAFETY_MARGIN:
        # If we are running out of time, fall back to a very fast, simple heuristic
        # or just return the mean-based score to ensure we finish.
        # Returning -mu is the most exploitative and fast option.
        return -mu
    
    # LCB parameters
    beta = 5.0
    
    # Ensure sigma is finite and non-negative
    sigma = np.asarray(sigma, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    
    # Clip sigma to avoid issues with zero variance
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate LCB score: maximize (-mu + beta * sigma)
    # This is equivalent to minimizing (mu - beta * sigma)
    scores = -mu + beta * sigma
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
