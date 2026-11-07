"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _call_count
    _call_count += 1
    
    # Check time limit to ensure we return within constraints
    if time.time() - _start_time > 320:
        # Fallback: pure exploitation
        return -mu

    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Use UCB (Upper Confidence Bound) for better exploration in early stages
    # The first few calls (initial random points + a few more) benefit from more exploration
    # After that, we can shift to exploitation
    
    # Dynamic beta: high exploration early, decreasing over time
    # With 30 total evaluations, we have 25 acquisition calls (after 5 initial points)
    # For the first 5-10 calls, use high beta; then decrease
    
    if _call_count <= 5:
        beta = 3.0  # Very high exploration for first few steps
    elif _call_count <= 10:
        beta = 2.0  # Still high exploration
    elif _call_count <= 15:
        beta = 1.5  # Moderate exploration
    elif _call_count <= 20:
        beta = 1.0  # Balanced
    else:
        beta = 0.5  # More exploitation in later stages
    
    # UCB for minimization: we want to maximize -mu + beta * sigma
    # This is equivalent to minimizing mu - beta * sigma
    # So we return -mu + beta * sigma as the score (higher is better)
    
    ucb = -mu + beta * sigma_safe
    
    # Ensure finite scores
    ucb[~np.isfinite(ucb)] = 0.0
    
    return ucb
# EVOLVE-BLOCK-END
