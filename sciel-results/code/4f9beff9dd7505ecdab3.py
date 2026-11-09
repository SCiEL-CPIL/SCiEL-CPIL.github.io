"""Acquisition function for Bayesian optimization of goldstein price 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points.
    Uses Thompson Sampling: score is the negative of a sample from the posterior.
    This implicitly balances exploration and exploitation without tuning a beta parameter
    and naturally handles the multimodal Goldstein-Price landscape.
    """
    # Deadline guard: track elapsed time to ensure we return before 300s
    t0 = time.time()
    deadline = 300.0 - 2.0  # safety margin
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    # Check deadline before heavy computation
    if time.time() > deadline:
        # Fallback: simple UCB
        return -mu + 1.0 * sigma
    
    # Thompson Sampling: sample y ~ N(mu, sigma)
    # Higher score means more likely to be a better (lower) value
    # So we use -y as the score (negative of the sampled objective value)
    
    # Generate random normal samples
    rng = np.random.default_rng()
    z = rng.standard_normal(mu.shape)
    sample = mu + sigma * z
    
    # Score is negative of the sample (we want to minimize, so higher score = better)
    scores = -sample
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Final deadline check
    if time.time() > deadline:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
