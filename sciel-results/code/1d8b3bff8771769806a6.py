"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Start time for deadline guard
_start_time = time.time()
DEADLINE_MARGIN = 10  # seconds

def acquisition(mu, sigma, f_best):
    """
    Lower Confidence Bound (LCB) acquisition function for minimization.
    The evaluator selects the candidate with the HIGHEST score.
    We want to minimize mu - beta * sigma.
    So we return -mu + beta * sigma.
    """
    # Check deadline
    if time.time() - _start_time > 330 - DEADLINE_MARGIN:
        # Fallback to pure exploitation if running out of time
        # Just return -mu to be safe and fast
        return -mu

    beta = 0.5
    # LCB for minimization: mu - beta * sigma
    # We want to maximize the score, so we return the negative of the LCB
    score = -mu + beta * sigma
    
    # Ensure finite scores
    # sigma should be positive, mu should be finite
    # If any non-finite values exist, replace them with very negative scores
    # so they are not selected.
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
