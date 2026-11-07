"""Rewritten acquisition for Bayesian optimization of sphere 1d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using LCB with zero-crossing likelihood term.
    
    Combines:
    1. LCB term: -mu + beta * sigma (exploitation/exploration balance)
    2. Zero-crossing term: gamma * Phi(-mu/sigma) (smooth attraction to f=0)
    
    The zero-crossing term creates a sigmoidal gradient that peaks at mu=0,
    directly targeting the theoretical minimum of the sphere function.
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # LCB component: -mu + beta * sigma
    beta = 0.15
    lcb_score = -mu + beta * sigma_safe
    
    # Zero-crossing likelihood: Phi(-mu/sigma)
    # This is high when mu is negative or close to 0 relative to sigma
    # For sphere function (f >= 0), this creates a smooth peak at mu=0
    z = -mu / sigma_safe
    zero_cross = norm.cdf(z)
    
    # Scale the zero-crossing term to be comparable to LCB scores
    # LCB scores range roughly from -5 to 5 (for mu in [-5,5], sigma ~ 0.1-1)
    # zero_cross ranges from 0 to 1
    gamma = 1.0
    
    score = lcb_score + gamma * zero_cross
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Time guard: if we've exceeded 2 seconds, return immediately
    elapsed = time.time() - start_time
    if elapsed > 2.0:
        return score
    
    return score
# EVOLVE-BLOCK-END
