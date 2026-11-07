"""Rewritten acquisition for Bayesian optimization of Hartmann 6d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: 330s wall-clock limit, use 325s to be safe
_DEADLINE = time.time() + 325.0


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Probability of Improvement (PI) with a
    normalized exploration term.

    Structural change: Replaces Expected Improvement (EI) with Probability
    of Improvement (PI). PI focuses on the likelihood of finding a point
    better than f_best, rather than the expected magnitude of improvement.
    This can lead to more diverse exploration in multimodal landscapes by
    not over-weighting points with high potential improvement magnitude but
    lower probability. A normalized sigma term is added to encourage
    exploration in high-uncertainty regions.

    Parameters
    ----------
    mu : ndarray, shape (n,)
        GP mean predictions.
    sigma : ndarray, shape (n,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed objective value so far.

    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values are preferred.
    """
    # Time check: if we're close to deadline, return a simple score
    if time.time() > _DEADLINE - 5.0:
        mu = np.asarray(mu, dtype=np.float64)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        return -mu

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Safety: replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma_safe = np.maximum(sigma, 1e-12)

    # Calculate z-score for Probability of Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe

    # Probability of Improvement: P(f(x) < f_best)
    pi = norm.cdf(z)

    # Normalize sigma to [0, 1] range relative to the batch
    # This makes the exploration term comparable to PI (which is in [0, 1])
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-12:
        sigma_norm = sigma_safe / (sigma_max + 1e-12)
    else:
        sigma_norm = np.zeros_like(sigma_safe)

    # Exploration weight
    # We use a fixed beta for simplicity, but could adapt it.
    # A beta of 1.0 means uncertainty is weighted equally to the probability of improvement.
    beta = 1.0
    
    scores = pi + beta * sigma_norm

    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
