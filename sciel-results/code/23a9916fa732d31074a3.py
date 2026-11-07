"""Acquisition function for Bayesian optimization of a fixed seeded random RBF mixture in 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _call_count
    _call_count += 1
    
    # Deadline guard: if we are too close to the limit, return a simple score
    if time.time() - _start_time > 325:
        return -mu

    t = _call_count
    n = len(mu)
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Structural Change: Use Expected Improvement (EI) for minimization.
    # EI(x) = E[max(f_best - f(x), 0)]
    # For a GP posterior N(mu, sigma^2), EI is:
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu) / sigma
    # Phi is the standard normal CDF and phi is the standard normal PDF.
    
    # Calculate Z scores
    # Handle division by zero safely
    z = (f_best - mu) / sigma_safe
    
    # Compute EI
    # Using scipy.stats.norm for CDF and PDF
    ei_scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure finite scores
    ei_scores = np.where(np.isfinite(ei_scores), ei_scores, 0.0)
    
    return ei_scores
# EVOLVE-BLOCK-END
