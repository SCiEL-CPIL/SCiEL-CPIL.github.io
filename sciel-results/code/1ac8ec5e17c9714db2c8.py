"""Cost-aware acquisition for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 295.0  # safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling with adaptive
    cost normalization.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions.
    sigma : array of shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far.
    cost : array of shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = more desirable).
    """
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        # Return a safe fallback: prefer low mu and low cost
        mu_safe = np.asarray(mu, dtype=np.float64)
        cost_safe = np.asarray(cost, dtype=np.float64)
        mu_safe = np.where(np.isfinite(mu_safe), mu_safe, 0.0)
        cost_safe = np.where(np.isfinite(cost_safe) & (cost_safe > 0), cost_safe, 1.0)
        return np.where(np.isfinite(-mu_safe / cost_safe), -mu_safe / cost_safe, 0.0)
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-6, 1e2)
    
    # Thompson Sampling: draw a single sample from the posterior
    # sample ~ N(mu, sigma^2)
    rng = np.random.default_rng()
    sample = mu + sigma * rng.standard_normal(len(mu))
    
    # Adaptive cost normalization: normalize cost relative to current best
    # This makes the cost penalty scale with the problem's cost landscape
    cost_mean = np.mean(cost)
    cost_std = np.std(cost)
    if cost_std > 1e-10:
        cost_z = (cost - cost_mean) / cost_std
    else:
        cost_z = np.zeros_like(cost)
    
    # Clip cost_z to avoid extreme values
    cost_z = np.clip(cost_z, -3.0, 3.0)
    
    # Cost-aware scoring: prefer low sampled value, penalize high cost
    # Score = -sample * exp(-alpha * cost_z) where alpha controls cost sensitivity
    # Using exponential to smoothly down-weight expensive candidates
    alpha = 0.5
    cost_weight = np.exp(-alpha * cost_z)
    
    # Combine: negative sampled value weighted by cost preference
    scores = -sample * cost_weight
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
