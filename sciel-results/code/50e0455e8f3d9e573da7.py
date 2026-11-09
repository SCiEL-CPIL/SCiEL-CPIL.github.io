"""Rewritten program for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_START_TIME = None
_DEADLINE_SECONDS = 315.0  # Safety margin before 330s kill


def _ensure_deadline():
    """Initialize or check the deadline."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME < _DEADLINE_SECONDS


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling for Bayesian optimization
    of Goldstein-Price 2D.
    
    Thompson Sampling draws a single sample from the posterior GP at each
    candidate point and selects the point with the highest sampled value.
    This provides intrinsic exploration-exploitation balance without manual
    tuning of uncertainty coefficients.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points.
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Thompson Sampling scores; higher means evaluate next.
    """
    # Check deadline
    if not _ensure_deadline():
        # Fallback: simple EI to be safe
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        return (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-9)
    
    # Thompson Sampling: draw one sample from N(mu, sigma^2) at each point
    # Use a fixed random seed per call for reproducibility within a seed run,
    # but the randomness comes from the sampling itself
    rng = np.random.default_rng()
    samples = mu + sigma_safe * rng.standard_normal(len(mu))
    
    # Ensure all scores are finite
    samples = np.where(np.isfinite(samples), samples, 0.0)
    
    return samples
# EVOLVE-BLOCK-END
