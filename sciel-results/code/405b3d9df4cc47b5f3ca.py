"""Rewritten acquisition for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: total wall-clock budget is 330 s; return safely before that.
_DEADLINE = 325.0
_START = time.time()
# Track number of acquisition calls to adapt exploration
_call_count = 0


def _elapsed():
    return time.time() - _START


def acquisition(mu, sigma, f_best):
    """
    Posterior-sampling acquisition approximating Expected Improvement.
    
    Draw 50 latent function samples from the GP posterior at the 2000 candidates.
    For each sample, compute the improvement over f_best (f_best - y_sample).
    The acquisition score is the mean of the maximum improvement across samples
    for each candidate, which approximates E[max(0, f_best - f(x))].
    """
    global _call_count
    _call_count += 1

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = len(mu)

    # Safety: replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)

    # Check deadline: if we're running out of time, return a simple score
    if _elapsed() > _DEADLINE - 5.0:
        # Fallback: use UCB-like score
        scores = -(mu - 0.1 * sigma)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Number of posterior samples for EI approximation
    n_samples = 50

    # Generate random normal samples
    rng = np.random.default_rng()

    # Posterior sampling: for each sample, draw y_i ~ N(mu_i, sigma_i^2)
    # Shape: (n_samples, n)
    samples = mu[None, :] + sigma[None, :] * rng.standard_normal((n_samples, n))

    # For each sample, compute improvement: f_best - y_sample
    # Only positive improvements count (max with 0)
    improvements = np.maximum(f_best - samples, 0.0)  # shape: (n_samples, n)

    # The acquisition score for each candidate is the mean improvement across samples
    # This approximates E[max(0, f_best - f(x))] = EI
    scores = improvements.mean(axis=0)  # shape: (n,)

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
