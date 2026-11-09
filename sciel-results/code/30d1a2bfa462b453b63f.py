"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state to track elapsed time for the deadline guard
_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    
    # Initialize start time on first call
    if _start_time is None:
        _start_time = time.time()
    
    n = len(mu)
    if n == 0:
        return np.array([])

    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-8)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Deadline guard: if we are running too long, return greedy scores (-mu)
    # to ensure we finish within the time limit.
    elapsed = time.time() - _start_time
    if elapsed > 290:
        # Greedy: pick the point with the lowest mean prediction
        # Score = -mu (higher is better)
        return -mu

    # Expected Improvement for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # where f(x) ~ N(mu(x), sigma(x)^2)
    
    # Standardized improvement: z = (f_best - mu) / sigma
    # EI = sigma * [z * Phi(z) + phi(z)]
    
    z = (f_best - mu) / sigma
    
    # Compute Phi(z) (CDF of standard normal) and phi(z) (PDF of standard normal)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI formula
    ei = sigma * (z * Phi_z + phi_z)
    
    # Ensure finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Higher EI means more expected improvement, so return EI as score
    return ei
# EVOLVE-BLOCK-END
