"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_start_time = time.time()
_DEADLINE = 290.0  # seconds, safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Samples from the Gaussian posterior for each candidate, averages multiple
    samples for stability, and divides by cost to balance exploration and
    exploitation while preferring cheaper evaluations.
    """
    # Deadline guard: if we're close to the limit, return a simple fallback
    if time.time() - _start_time > _DEADLINE:
        # Fallback: use expected value (mu) / cost
        cost_safe = np.maximum(cost, 1e-10)
        scores = -mu / cost_safe  # lower mu is better, so negative
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Thompson Sampling: sample from N(mu, sigma) for each candidate
    # Use multiple samples to reduce variance
    n_samples = 5
    samples = mu + sigma_safe * np.random.standard_normal((n_samples, len(mu)))
    
    # Average the samples for a more stable estimate
    ts_score = np.mean(samples, axis=0)
    
    # Cost-aware: prefer lower objective value per unit cost
    # We want to minimize f, so we use -ts_score / cost (higher is better)
    scores = -ts_score / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
