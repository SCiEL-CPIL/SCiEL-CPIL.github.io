"""Acquisition function for Bayesian optimization of goldstein price 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores using Thompson Sampling.
    For each candidate, sample y ~ N(mu, sigma) and return -y as the score.
    Higher scores correspond to lower sampled values, balancing exploration
    (high sigma regions) and exploitation (low mu regions) automatically.
    """
    # Deadline guard
    t0 = time.time()
    deadline = 300.0 - 2.0  # safety margin
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    # Check deadline before computation
    if time.time() > deadline:
        return -mu
    
    # Thompson Sampling: sample from the posterior at each candidate
    # y_i ~ N(mu_i, sigma_i^2)
    # Score is -y_i (higher is better, meaning lower predicted value)
    rng = np.random.default_rng()
    samples = mu + sigma * rng.standard_normal(len(mu))
    
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
