"""Acquisition function for Bayesian optimization of branin 2d using Thompson Sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    Uses Thompson Sampling: draw a single sample from the posterior
    at each candidate point and return the negative sample as the score.
    Higher scores (more negative samples) are preferred, which naturally
    balances exploration and exploitation through the randomness of the samples.
    """
    # Time budget guard
    elapsed = time.time() - _start_time
    if elapsed > 290:
        # Fallback: simple EI
        sigma_safe = np.where(np.asarray(sigma, dtype=np.float64) > 1e-12,
                              np.asarray(sigma, dtype=np.float64), 1e-12)
        mu_safe = np.asarray(mu, dtype=np.float64)
        z = (f_best - mu_safe) / sigma_safe
        ei = (f_best - mu_safe) * norm_cdf(z) + sigma_safe * norm_pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma = np.where(sigma > 1e-12, sigma, 1e-12)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Thompson Sampling: sample from N(mu, sigma^2) at each candidate
    # Use a fixed random state for reproducibility within a single call,
    # but the seed changes across calls since we use the global RNG
    rng = np.random.default_rng()
    samples = mu + sigma * rng.standard_normal(len(mu))
    
    # Return negative samples as scores (higher = better = lower function value)
    scores = -samples
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

def norm_cdf(x):
    """Standard normal CDF."""
    return 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))

def norm_pdf(x):
    """Standard normal PDF."""
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)

def _erf(x):
    """Approximation of the error function."""
    # Abramowitz and Stegun approximation
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    
    sign = np.sign(x)
    x = np.abs(x)
    
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    
    return sign * y
# EVOLVE-BLOCK-END
