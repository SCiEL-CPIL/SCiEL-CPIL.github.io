"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_deadline = _start_time + 290.0

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() > _deadline:
        return -mu
    
    # Sanitize inputs
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = np.maximum(sigma, 1e-10)
    sigma = np.nan_to_num(sigma, nan=1e-10, posinf=1e-10, neginf=1e-10)
    
    # Expected Improvement (EI) for minimization
    # We want to maximize the expected reduction in f_best.
    # EI(x) = E[max(0, f_best - f(x))]
    
    # z-score
    z = (f_best - mu) / sigma
    # Clip z to prevent numerical issues in exp
    z = np.clip(z, -50.0, 50.0)
    
    # Standard Normal PDF and CDF
    pdf = norm.pdf(z)
    cdf = norm.cdf(z)
    
    # EI formula
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Ensure finite scores
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return ei
# EVOLVE-BLOCK-END
