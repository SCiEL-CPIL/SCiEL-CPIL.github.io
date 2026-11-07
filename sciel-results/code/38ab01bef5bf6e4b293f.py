"""Acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    Structural change: Sample a single function value from the predictive
    distribution N(mu, sigma^2) at each of the 2000 candidates and return
    the negative sampled value as the score (higher = better = lower f).
    This inherently balances exploration and exploitation without requiring
    a heuristic combination of mean and variance terms.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    t_start = time.time()
    
    # Safety: clip sigma to avoid zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Thompson Sampling: draw one sample from each N(mu_i, sigma_i^2)
    # Use a single standard normal draw and scale
    z = np.random.randn(n)
    samples = mu + sigma_safe * z  # shape (n,)
    
    # Score = negative sampled value (higher score = lower f = better)
    scores = -samples
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Deadline guard
    if time.time() - t_start > 280:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
