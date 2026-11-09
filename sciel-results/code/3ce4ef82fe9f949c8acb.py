"""Rewritten acquisition function for 1D sphere Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1D on [-5, 5].
    
    Uses a pure Expected Improvement (EI) acquisition, which is well-suited
    for this smooth, single-objective problem. EI naturally balances exploration
    and exploitation without requiring tuning of a kappa parameter.
    
    Args:
        mu: array of shape (2000,) - GP posterior mean at candidate points
        sigma: array of shape (2000,) - GP posterior std at candidate points
        f_best: float - best (lowest) observed value so far
    
    Returns:
        scores: array of shape (2000,) - acquisition scores (higher = more promising)
    """
    # Deadline guard to prevent timeout
    if not hasattr(acquisition, '_start_time'):
        acquisition._start_time = time.time()
    
    elapsed = time.time() - acquisition._start_time
    if elapsed > 290:
        # Fast fallback: simple PI
        z = (f_best - mu) / (sigma + 1e-9)
        return norm.cdf(z)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # For minimization: improvement is f_best - f(x), so we want f(x) < f_best
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid extreme values causing numerical issues
    z = np.clip(z, -50, 50)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure all scores are finite
    scores = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
