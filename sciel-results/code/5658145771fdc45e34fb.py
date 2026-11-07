"""Acquisition function for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard
_DEADLINE = time.time() + 280.0

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for minimization of Michalewicz 2D.
    
    This uses a stochastic approach: draw a single sample from the posterior
    f_s = mu + sigma * xi, where xi ~ N(0,1), and return -f_s as the score.
    Higher scores (lower sampled values) are selected next.
    
    This fundamentally differs from deterministic EI/LCB by introducing
    randomization that naturally balances exploration and exploitation.
    """
    # Deadline check
    if time.time() > _DEADLINE:
        # Fallback: return simple LCB-like scores
        return -mu + 2.0 * sigma
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and not too small
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw a single random sample from the posterior
    # xi ~ N(0, 1)
    xi = np.random.randn(len(mu))
    
    # Sampled function value: f_s = mu + sigma * xi
    f_sample = mu + sigma_safe * xi
    
    # Return -f_sample as the score (higher is better for selection)
    scores = -f_sample
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
