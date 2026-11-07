"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global variable to track start time for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    
    # Initialize start time on first call
    if _start_time is None:
        _start_time = time.time()
        
    # Check deadline (330s limit, leave 30s margin)
    # This is a safety net, the computation itself is fast
    if time.time() - _start_time > 300:
        # Fallback to a simple acquisition if time is running out
        # Return -mu (exploit only)
        return -mu

    # Adaptive beta for LCB
    # Higher f_best (worse) -> Higher beta (more exploration)
    # Lower f_best (better) -> Lower beta (more exploitation)
    # f_best is the minimum value observed so far.
    # For sphere 1d, f_best ranges from ~25 down to 0.
    
    # Heuristic:
    # If f_best is large, we want to explore widely.
    # If f_best is small, we want to exploit the neighborhood.
    
    # Let's use a smooth transition.
    # beta = 0.5 + 4.5 * (f_best / (f_best + 1.0))
    
    # Ensure f_best is positive (it should be for x^2)
    f_best_val = max(f_best, 1e-6)
    
    beta = 0.5 + 4.5 * (f_best_val / (f_best_val + 1.0))
    
    # LCB score: -mu + beta * sigma
    # We want to maximize this score.
    # High sigma increases score (exploration).
    # Low mu increases score (exploitation).
    
    score = beta * sigma - mu
    
    return score
# EVOLVE-BLOCK-END
