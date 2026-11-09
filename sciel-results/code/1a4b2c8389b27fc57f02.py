"""Rewritten acquisition for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: total wall-clock budget is 330 s; return safely before that.
_DEADLINE = 325.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points.
    Uses posterior sampling (Thompson sampling) to approximate Expected Improvement.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Safety: replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)

    n_samples = 50
    rng = np.random.default_rng()

    # Draw samples from the GP posterior
    # y_sample ~ N(mu, sigma^2)
    # shape: (n_samples, 2000)
    std_normal = rng.standard_normal((n_samples, mu.size))
    samples = mu[None, :] + sigma[None, :] * std_normal

    # For each sample, compute improvement over f_best: max(f_best - y, 0)
    improvements = np.maximum(f_best - samples, 0.0)

    # Acquisition score is the mean improvement over samples
    scores = np.mean(improvements, axis=0)

    # Check deadline: if we're running out of time, just return a simple score
    if _elapsed() > _DEADLINE - 5.0:
        # Fallback: exploit (lowest mu) with tiny exploration
        scores = -mu + 0.1 * sigma
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
