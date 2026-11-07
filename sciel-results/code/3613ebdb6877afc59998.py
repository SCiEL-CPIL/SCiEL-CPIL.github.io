"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, use 325s to be safe
_DEADLINE = time.time() + 325.0

def acquisition(mu, sigma, f_best):
    # Time check: if we're close to deadline, return a simple score
    if time.time() > _DEADLINE - 5.0:
        return -mu
    
    # Expected Improvement with small xi for slight exploration
    xi = 0.0
    z = (f_best - mu - xi) / (sigma + 1e-9)
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    return ei
# EVOLVE-BLOCK-END
