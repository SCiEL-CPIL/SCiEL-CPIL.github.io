"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard state (optional, but requested)
_start_time = None

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def acquisition(mu, sigma, f_best):
    # Ensure we have not exceeded a reasonable computation time
    # Although 2000 points is very fast, we guard against pathological cases
    if _get_elapsed() > 300:
         # Fallback to greedy if we are running out of time (unlikely)
         return -mu

    # Quantile-based acquisition function (Optimistic Minimization)
    # We want to minimize f(x).
    # The 5th percentile of N(mu, sigma^2) is mu - 1.645 * sigma.
    # We want to find the x with the lowest 5th percentile value.
    # The interface requires returning scores where HIGHER is better.
    # So we return the negative of the 5th percentile value.
    # Score = -(mu - 1.645 * sigma) = -mu + 1.645 * sigma
    
    beta = 1.645
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    # If sigma is 0, scores are -mu.
    # If mu/sigma are inf/nan, we need to handle it.
    # The problem states mu and sigma are from GP, should be finite.
    # But just in case:
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
