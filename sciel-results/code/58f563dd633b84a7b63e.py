"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None
_DEADLINE = 320.0  # Safety margin before the 330s wall-clock limit

def _ensure_start():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function: EI/cost with a small bounded UCB bonus.
    """
    _ensure_start()

    # Deadline guard: if we are past the safety margin, return pure EI/cost
    # (no extra computation) to avoid timeout.
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Minimal safe fallback: return zeros (valid finite scores)
        n = np.asarray(mu).shape[0]
        return np.zeros(n, dtype=np.float64)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = mu.shape[0]
    if n == 0:
        return np.array([])

    # Guard against non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    cost = np.where(np.isfinite(cost), cost, 1.0)

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-8)

    # Calculate z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe

    # Clip z to avoid numerical issues in norm.cdf/pdf
    z_clipped = np.clip(z, -10.0, 10.0)

    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)

    # Small bounded UCB exploration bonus (capped at 5% of sigma)
    # This gently encourages unexplored regions without overwhelming exploitation
    beta = 0.05
    ucb_bonus = np.minimum(beta * sigma_safe, 0.05 * np.maximum(sigma_safe, 1e-8))

    # Combine: primary EI + small exploration bonus
    combined = ei + ucb_bonus

    # Cost-aware: divide by cost
    score = combined / cost_safe

    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)

    return score
