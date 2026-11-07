import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 280.0
_CALL_COUNT = 0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on hartmann 3d.
    Uses a cost-effective UCB with adaptive beta that shifts from exploration
    to exploitation as the budget is consumed, combined with a smooth
    multiplicative cost discount.
    """
    global _CALL_COUNT
    _CALL_COUNT += 1

    # Deadline guard
    if (time.time() - _START_TIME) > _TIME_LIMIT:
        return -mu.copy()

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)

    # Compute Expected Improvement for minimization
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Adaptive beta: starts high for exploration, decays as budget is consumed
    # Budget is 30 total cost, starting with ~5 points (cost ~1.5 each = 7.5)
    # Remaining budget estimate: 30 - 1.5 * _CALL_COUNT
    # Normalize to [0, 1] fraction remaining
    est_remaining = max(30.0 - 1.5 * _CALL_COUNT, 1.0)
    frac_remaining = est_remaining / 30.0  # ~1.0 early, ~0.03 late

    # Beta decays from ~2.0 to ~0.1
    beta = 0.1 + 1.9 * frac_remaining ** 1.5

    # UCB score
    ucb_score = -mu + np.sqrt(beta) * sigma

    # Combine UCB and EI with smooth weighting
    # Early: more UCB weight; Late: more EI weight
    w_ucb = 0.5 * frac_remaining + 0.1
    w_ei = 1.0 - w_ucb

    base_score = w_ucb * ucb_score + w_ei * ei

    # Cost-aware discount: prefer cheaper points
    # Cost is in [1, 2]. Use a multiplicative discount that's smooth.
    # Normalize cost to [0, 1] range
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    if cost_max - cost_min > 1e-12:
        cost_norm = (cost - cost_min) / (cost_max - cost_min)
    else:
        cost_norm = np.zeros_like(cost)

    # Discount factor: ranges from 1.0 (cheapest) to ~0.7 (most expensive)
    # The strength of the discount increases as we approach the budget limit
    discount_strength = 0.3 * (1.0 - frac_remaining)
    discount = 1.0 - discount_strength * cost_norm

    score = base_score * discount

    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)

    return score
