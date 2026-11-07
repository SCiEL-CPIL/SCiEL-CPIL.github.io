"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
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
        from scipy.stats import norm
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        return (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Structural Change: Decaying UCB (Upper Confidence Bound for minimization)
    # We want to minimize f. 
    # LCB = mu - beta * sigma
    # We want to pick the point with the lowest LCB.
    # The acquisition function returns scores where HIGHER is better.
    # So we return -LCB = -mu + beta * sigma
    
    # Determine beta based on call count
    # Total 25 acquisition calls (30 evals - 5 initial)
    # Early calls: high exploration (high beta)
    # Late calls: high exploitation (low beta)
    
    if _call_count <= 3:
        beta = 4.0
    elif _call_count <= 6:
        beta = 3.0
    elif _call_count <= 10:
        beta = 2.0
    elif _call_count <= 15:
        beta = 1.5
    elif _call_count <= 20:
        beta = 1.0
    else:
        beta = 0.5
    
    # Calculate scores
    # sigma is standard deviation, already non-negative
    # Ensure sigma is finite and positive to avoid issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    scores = -mu + beta * sigma_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
