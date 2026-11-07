"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_call_count = 0
_start_time = None

def acquisition(mu, sigma, f_best):
    global _call_count, _start_time
    
    if _start_time is None:
        _start_time = time.time()
    _call_count += 1
    
    # Deadline guard: if we're close to the 330s limit, return a safe fallback
    elapsed = time.time() - _start_time
    if elapsed > 325:
        # Return a simple EI to finish quickly
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB with adaptive beta: high exploration early, decay over time
    # For Branin 2D with 25 evals, beta starts at ~4 and decays
    # We have 25 total evaluations, 5 initial, so 20 acquisition calls
    # Decay beta from 4.0 to 1.5 over the 20 calls
    if _call_count <= 5:
        beta = 4.0
    elif _call_count <= 10:
        beta = 3.0
    elif _call_count <= 15:
        beta = 2.0
    else:
        beta = 1.5
    
    # UCB: mu + beta * sigma
    # But we want higher = more interesting, and we minimize f
    # So we want to pick points where mu is low AND sigma is high
    # UCB for minimization: -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
