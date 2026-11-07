import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 280.0  # Leave 50s for cleanup/other overhead
_CALL_COUNT = 0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on hartmann 3d.
    Uses Cost-Effective Expected Improvement (CEI) combined with UCB exploration,
    where the cost adjustment is principled: expected improvement per unit cost.
    """
    global _CALL_COUNT
    _CALL_COUNT += 1

    # Deadline guard
    if (time.time() - _START_TIME) > _TIME_LIMIT:
        # Return a safe, fast score: exploit current best
        return -mu.copy()

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)

    # Compute Expected Improvement for minimization
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # UCB-style score: -mu + sqrt(beta) * sigma
    # beta decays more gradually to maintain exploration
    beta = 1.5 / (1.0 + 0.03 * _CALL_COUNT)
    ucb_score = -mu + np.sqrt(beta) * sigma

    # Cost-Effective EI: divide EI by cost to get improvement per unit cost
    # This naturally prefers cheaper points with similar expected improvement
    cei = ei / np.maximum(cost, 1e-9)

    # Combine CEI and UCB with cost-aware weighting
    # Early: more UCB exploration; Late: more CEI exploitation
    w_ucb = 0.6 / (1.0 + 0.08 * _CALL_COUNT)
    w_cei = 1.0 - w_ucb

    # Cost-adjusted UCB: also penalize expensive points in exploration
    # Use a soft cost penalty that's smooth
    cost_penalty_weight = 0.3 / (1.0 + 0.05 * _CALL_COUNT)
    cost_adj_ucb = ucb_score * (1.0 - cost_penalty_weight * (cost - 1.0))

    # Base score (higher is better)
    base_score = w_ucb * cost_adj_ucb + w_cei * cei

    # Ensure all scores are finite
    score = np.nan_to_num(base_score, nan=0.0, posinf=1e6, neginf=-1e6)

    return score
