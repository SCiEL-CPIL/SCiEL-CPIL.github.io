"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 280  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best):
    # Time guard: if we're close to the deadline, return pure EI (exploitation-focused)
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    sigma = np.maximum(sigma, 1e-9)
    
    # Use a small, fixed reference offset to mildly encourage exploration
    # The key insight: too much exploration hurts on Branin (multimodal but bounded)
    # Use a small fraction of the median sigma to keep reference close to f_best
    med_sigma = np.median(sigma)
    
    # Very modest reference point: just slightly worse than f_best
    # This makes EHVI only slightly more exploratory than standard EI
    alpha = 0.1
    f_ref = f_best + alpha * med_sigma
    
    # EHVI computation
    z_hv = (f_ref - mu) / sigma
    ehvi = (f_ref - mu) * norm.cdf(z_hv) + sigma * norm.pdf(z_hv)
    
    scores = ehvi
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
