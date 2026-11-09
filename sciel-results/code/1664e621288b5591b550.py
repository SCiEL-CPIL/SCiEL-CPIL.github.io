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
    Uses Analytical Expected Improvement (EI) for minimization.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Safety: replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)

    # Check deadline: if we're running out of time, just return a simple score
    if _elapsed() > _DEADLINE - 5.0:
        # Fallback: exploit (lowest mu)
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Analytical Expected Improvement for minimization
    # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x)) / sigma(x)) + sigma(x) * phi((f_best - mu(x)) / sigma(x))
    
    # Handle division by zero (sigma is clamped to 1e-9, so safe)
    z = (f_best - mu) / sigma
    
    # Clip z to avoid overflow in exp for large negative/positive values
    # Phi and phi are stable, but let's be safe
    z = np.clip(z, -10, 10)
    
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    improvement = f_best - mu
    ei = improvement * cdf + sigma * pdf
    
    # Ensure all scores are finite
    scores = ei
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
