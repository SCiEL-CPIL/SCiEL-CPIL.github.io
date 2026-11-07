"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level timer for the deadline guard
_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Lower Confidence Bound (LCB) for minimization.
    Score = -mu + kappa * sigma
    Higher scores are preferred.
    
    We use a slightly higher kappa (1.5) compared to the baseline (1.0) to encourage
    more exploration, as the Ackley function has many local minima and the GP
    with length scale 0.2 may need to explore broader regions.
    """
    # Deadline guard: Check if we are close to the 330s limit.
    # If so, return a safe, exploitation-heavy score to ensure we finish.
    elapsed = time.time() - _start_time
    if elapsed > 320.0:
        # Pure exploitation: pick lowest mean
        # Ensure finite
        mu_safe = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
        return -mu_safe

    # LCB parameter
    # Baseline was 1.0. Increasing to 1.5 to boost exploration.
    kappa = 1.5
    
    # Compute scores
    # We want high scores for low mu and high sigma
    scores = -mu + kappa * sigma
    
    # Ensure finite scores
    # If sigma is NaN or Inf, handle it
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
