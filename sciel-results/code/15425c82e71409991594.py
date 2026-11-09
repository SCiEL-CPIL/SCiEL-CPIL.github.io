import numpy as np
from scipy.stats import norm
import time

_start_time = None
_call_count = 0
_total_evals = 0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Shekel 4D.
    Uses a hybrid approach: cost-aware EI with adaptive exploration and a
    probability-of-improvement boost for high-confidence improvements.
    """
    global _start_time, _call_count, _total_evals

    if _start_time is None:
        _start_time = time.time()
    _call_count += 1
    _total_evals += 1

    elapsed = time.time() - _start_time
    if elapsed > 315:
        # Safety fallback: simple cost-aware exploitation
        return -mu / np.maximum(cost, 1e-9)

    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-12, 1e6)
    cost_safe = np.maximum(cost, 1e-9)

    # Calculate z-score for EI
    z = (f_best - mu) / sigma

    # Expected Improvement (EI) for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Cost-aware EI
    cost_aware_ei = ei / cost_safe

    # Adaptive exploration based on total evaluations
    # In early iterations, explore more; in later iterations, exploit more
    if _total_evals <= 10:
        exploration_factor = 0.08
    elif _total_evals <= 25:
        exploration_factor = 0.04
    elif _total_evals <= 50:
        exploration_factor = 0.02
    else:
        exploration_factor = 0.01

    # Upper Confidence Bound (UCB) style term
    ucb_term = sigma / cost_safe * exploration_factor

    # Combine EI and exploration
    scores = cost_aware_ei + ucb_term

    # Probability of improvement (PI) boost
    pi = norm.cdf(z)

    # Add a small bonus for high probability of improvement
    scores = scores * (1.0 + 0.2 * pi)

    # Add a small penalty for very high cost to encourage cheaper evaluations
    cost_penalty = np.where(cost_safe > 1.5, 0.08 * (cost_safe - 1.5), 0.0)
    scores = scores - cost_penalty

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
