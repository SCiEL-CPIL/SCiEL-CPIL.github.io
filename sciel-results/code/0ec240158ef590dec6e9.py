"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None


def acquisition(mu, sigma, f_best):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    n = len(mu)
    elapsed = time.time() - _START_TIME
    # Deadline guard: if we're close to timeout, return simple EI as fallback
    if elapsed > 280:
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Monte Carlo acquisition: sample from posterior and compute probability of improvement
    # This is a structural change from deterministic EI to stochastic sampling-based scoring
    
    # Use adaptive number of samples based on remaining time budget
    base_samples = 50
    remaining = 300 - elapsed
    if remaining < 30:
        n_samples = 10
    elif remaining < 60:
        n_samples = 25
    else:
        n_samples = base_samples
    
    # Sample from the Gaussian posterior: x ~ N(mu, sigma^2)
    # For each sample, compute probability that it beats f_best
    # Score = E[1(x < f_best)] * (f_best - x) + exploration bonus
    
    # Vectorized Monte Carlo
    rng = np.random.RandomState(42)
    
    # Generate samples: shape (n_samples, n)
    samples = mu[None, :] + sigma[None, :] * rng.standard_normal((n_samples, n))
    
    # For each candidate, compute the expected improvement via Monte Carlo
    # EI_i = E[max(f_best - x_i, 0)] where x_i ~ N(mu_i, sigma_i^2)
    improvements = np.maximum(f_best - samples, 0.0)
    ei_mc = improvements.mean(axis=0)
    
    # Add a small exploration bonus for high uncertainty to encourage exploration
    # This helps escape local optima
    exploration_weight = 0.1
    exploration_bonus = exploration_weight * sigma
    
    # Combine: primary signal is MC-EI, secondary is exploration
    scores = ei_mc + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
