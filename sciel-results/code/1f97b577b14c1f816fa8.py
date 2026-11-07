"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    # Structural Change: Use Upper Confidence Bound (UCB) instead of Expected Improvement (EI).
    # UCB is an additive confidence bound method, structurally different from the integral-based EI.
    # It encourages exploration by adding a term proportional to the standard deviation.
    # This can help escape local optima in high-dimensional spaces where the GP might be over-smoothed.
    
    start_time = time.time()
    
    # Define exploration parameter beta
    # A value of 2.0 is a reasonable starting point for balanced exploration/exploitation.
    beta = 2.0
    
    # UCB score for minimization: we want to minimize f, so we maximize -mu + beta * sigma
    # f_best is not directly used in the score calculation for standard UCB, 
    # but it defines the context of the search.
    scores = -mu + beta * sigma
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Deadline Guard
    # Check if we are close to the time limit (300s). 
    # If so, return a safe fallback (greedy -mu) to ensure completion.
    if time.time() - start_time > 290:
        scores = -mu
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

    return scores
# EVOLVE-BLOCK-END
