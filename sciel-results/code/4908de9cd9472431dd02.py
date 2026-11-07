"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Use UCB (Lower Confidence Bound for minimization) per unit cost
    # We want to minimize f, so we look for low mu and high sigma.
    # Score to maximize: (-mu + beta * sigma) / cost
    beta = 1.5
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Time guard
    if time.time() - start_time > 290.0:
        # Fallback: simple exploitation
        return -mu / cost

    ucb_score = -mu + beta * sigma_safe
    scores = ucb_score / cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
