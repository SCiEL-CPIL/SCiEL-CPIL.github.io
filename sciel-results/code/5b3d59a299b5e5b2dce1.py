"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: kill after 330s wall-clock, keep a safety margin
_DEADLINE = 330.0
_START_TIME = time.time()
_TIME_MARGIN = 5.0  # seconds to spare


def _elapsed():
    return time.time() - _START_TIME


def _check_deadline():
    """Return True if we should stop (near deadline)."""
    return _elapsed() > (_DEADLINE - _TIME_MARGIN)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - evaluation costs
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Deadline check: if we're running out of time, return a simple fallback
    if _check_deadline():
        return (f_best - mu) / (cost + 1e-9)

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)

    # Compute normalized z-scores
    z = (f_best - mu) / sigma

    # Clip z to avoid numerical issues in CDF/PDF
    z = np.clip(z, -10, 10)

    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Also compute Probability of Improvement (PI) as a secondary signal
    pi = norm.cdf(z)

    # Lower bound on predicted value (optimistic)
    lb = mu - 2.0 * sigma
    improvement_lb = np.maximum(f_best - lb, 0.0)

    # Combine EI with a penalty for high cost
    # Use a softer cost penalty than quadratic: cost^1.5
    cost_penalty = cost ** 1.5

    # Primary score: EI / cost_penalty
    score_ei = ei / cost_penalty

    # Secondary score: PI / cost (for exploration)
    score_pi = pi / cost

    # Tertiary: lower-bound improvement / cost (very aggressive exploration)
    score_lb = improvement_lb / cost

    # Weighted combination: emphasize EI but include exploration components
    # This helps escape local minima in the acquisition landscape
    score = 0.7 * score_ei + 0.2 * score_pi + 0.1 * score_lb

    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
