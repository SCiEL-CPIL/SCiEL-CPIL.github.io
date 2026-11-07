import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 280.0  # Leave 50s for cleanup/other overhead
_CALL_COUNT = 0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on hartmann 3d.
    Combines UCB-style exploration with EI-style exploitation, using a global
    budget-aware cost penalty that shifts the exploration-exploitation balance
    based on remaining budget (estimated from call count).
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
    # beta decays with call count to shift from exploration to exploitation
    beta = 2.0 / (1.0 + 0.05 * _CALL_COUNT)
    ucb_score = -mu + np.sqrt(beta) * sigma

    # Combine EI and UCB: use a weighted sum
    # Early on, UCB helps explore; later, EI dominates for exploitation
    # Weight shifts from UCB to EI as we progress
    w_ucb = 1.0 / (1.0 + 0.1 * _CALL_COUNT)
    w_ei = 1.0 - w_ucb

    # Base score (higher is better)
    base_score = w_ucb * ucb_score + w_ei * ei

    # Global budget-aware cost penalty
    # Cost is in [1.0, 2.0]. We want to prefer lower cost.
    # Use a smooth penalty that's zero at mean cost and negative for higher cost.
    mean_cost = np.mean(cost)
    if mean_cost > 1e-12:
        cost_ratio = cost / mean_cost
        # Penalty: subtract a fraction of the base score for expensive points
        # Use a smooth function that penalizes cost > mean_cost
        excess_cost = np.maximum(cost_ratio - 1.0, 0.0)
        # Scale the penalty by the magnitude of the base score
        base_scale = np.mean(np.abs(base_score))
        if base_scale > 1e-12:
            penalty = 0.5 * excess_cost * base_scale
        else:
            penalty = np.zeros_like(cost)
    else:
        penalty = np.zeros_like(cost)

    score = base_score - penalty

    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)

    return score
