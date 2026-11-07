"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
# Safety margin: The limit is 330s. We aim to finish within 320s to be safe.
_DEADLINE = _START_TIME + 320.0

def acquisition(mu, sigma, f_best):
    # Explicit deadline guard
    if time.time() > _DEADLINE:
        # Fallback: Greedy selection based on mean
        # Return -mu so that the max score corresponds to the min mu
        return -mu

    # Ensure sigma is positive to avoid division by zero issues
    sigma_safe = np.maximum(sigma, 1e-8)

    # Lower Confidence Bound (LCB)
    # We want to minimize f. The LCB is mu - beta * sigma.
    # We return a score where higher is better.
    # So we return -LCB = -mu + beta * sigma.
    
    # Adaptive beta:
    # We can use a constant beta or an adaptive one.
    # A common choice for branin is beta around 2-5.
    # Let's try a slightly higher beta to encourage exploration initially, 
    # but since we don't have the iteration count, we stick to a robust constant.
    # The "Hypervolume-proxy" inspiration suggested structural favoring of high uncertainty.
    # LCB is a standard robust choice. Let's use beta = 3.0.
    
    beta = 3.0
    score = -mu + beta * sigma_safe

    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
