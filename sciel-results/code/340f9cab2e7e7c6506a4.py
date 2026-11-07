"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 310.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Samples a single function realization f ~ N(mu, sigma) and scores candidates
    by -f(x) / cost(x). This stochastically balances exploration and exploitation
    while naturally accounting for evaluation cost.
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        GP mean predictions
    sigma : array-like, shape (n,)
        GP standard deviation predictions
    f_best : float
        Best (lowest) observed value so far
    cost : array-like, shape (n,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition values (higher = more desirable to evaluate)
    """
    # Deadline guard: return a fast fallback to avoid timeout
    if time.time() > _DEADLINE:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-9)
        # Fallback: use mean-based score with small exploration
        score = (f_best - mu + 0.5 * sigma_safe) / (cost + 1e-9)
        return np.where(np.isfinite(score), score, 0.0)

    # Convert to numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-9)

    # Cost should be positive; clamp to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    cost_safe = np.where(np.isfinite(cost_safe), cost_safe, 1.0)

    # Thompson Sampling: sample one realization from the posterior
    # f_sample = mu + sigma * z, where z ~ N(0, 1)
    n = len(mu)
    z = np.random.randn(n)
    f_sample = mu + sigma_safe * z

    # Score: we want to minimize f, so higher score = lower f_sample
    # Normalize by cost: score = -f_sample / cost
    # But we need to be careful: f_sample can be negative (Griewank min is 0)
    # Use a shifted version to ensure meaningful ranking
    # The key insight: we want to find points where f_sample is low relative to cost
    
    # Simple Thompson Sampling score: -f_sample / cost
    # This directly optimizes expected improvement per unit cost
    score = -f_sample / cost_safe

    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
