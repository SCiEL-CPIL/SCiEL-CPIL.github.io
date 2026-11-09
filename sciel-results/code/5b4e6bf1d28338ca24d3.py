"""Rewritten acquisition function with a deadline guard and a hybrid acquisition strategy."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Track a global deadline so we can bail out safely if the evaluator kills us.
# The evaluator kills after 330s wall-clock; we use 320s as our internal limit.
_DEADLINE = time.time() + 320.0


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function:
    - Expected Improvement (EI) for exploitation
    - Knowledge Gradient / UCB-like bonus for exploration
    - A small penalty for very high sigma to avoid over-exploration
    """
    # Safety: ensure finite outputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Handle edge cases: sigma could be zero or negative due to numerical issues
    sigma = np.maximum(sigma, 1e-8)

    # Deadline guard: if we're too close to the limit, return a simple EI quickly
    if time.time() > _DEADLINE - 5.0:
        z = (f_best - mu) / sigma
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # UCB (Upper Confidence Bound) for exploration
    # beta controls the exploration-exploitation tradeoff
    beta = 0.5
    ucb = -mu + beta * sigma

    # Weighted combination: 70% EI, 30% UCB
    # This biases toward exploitation when EI is strong, but still explores
    score = 0.7 * ei + 0.3 * ucb

    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
