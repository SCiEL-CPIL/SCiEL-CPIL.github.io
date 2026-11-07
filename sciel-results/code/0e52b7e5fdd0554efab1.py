"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_call_count = 0
_start_time = None

def acquisition(mu, sigma, f_best):
    global _call_count, _start_time
    
    if _start_time is None:
        _start_time = time.time()
    _call_count += 1
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 325:
        return -mu

    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Hypervolume-based acquisition: Expected Hypervolume Improvement (EHVI)
    # For single objective, EHVI with reference point r is the expected improvement
    # relative to r.
    # r = f_best - delta
    # If we use r = f_best, it is standard EI.
    # Using a slightly lower r encourages finding significantly better points.
    # We use a small delta to avoid being too greedy.
    
    # Adaptive delta: start with a larger exploration gap, reduce it later
    if _call_count <= 5:
        delta = 0.05
    elif _call_count <= 10:
        delta = 0.02
    else:
        delta = 0.005
        
    f_ref = f_best - delta
    
    # Calculate EI relative to f_ref
    # If mu > f_ref, improvement is (f_ref - mu)
    # z = (f_ref - mu) / sigma
    z = (f_ref - mu) / sigma
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ehvi = (f_ref - mu) * Phi + sigma * phi
    
    # If f_ref is very low, ehvi might be negative or zero for many points.
    # We want to maximize this.
    # Ensure finite
    ehvi = np.where(np.isfinite(ehvi), ehvi, -np.inf)
    
    return ehvi
# EVOLVE-BLOCK-END
