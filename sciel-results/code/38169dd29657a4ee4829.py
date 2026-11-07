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
        
    # Expected improvement.
    # Use a slightly larger epsilon for stability if sigma is very small
    z = (f_best - mu) / (sigma + 1e-8)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure finite values
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return ei
# EVOLVE-BLOCK-END
