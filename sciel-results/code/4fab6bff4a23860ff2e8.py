"""Cost-aware acquisition for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard: track when the process started.
# The evaluator kills at 330s wall-clock; we use a safety margin.
_PROCESS_START = time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the 330s kill to bail out safely
_DEADLINE = _PROCESS_START + 330.0 - _DEADLINE_MARGIN


def _time_remaining():
    return _DEADLINE - time.time()


def acquisition(mu, sigma, f_best, cost):
    """
    Adaptive cost-aware Expected Improvement.

    Strategy:
      - Early phase (low total cost): use pure EI (ignore cost) to explore
        freely and build a good GP model.
      - Late phase (high total cost): use cost-aware EI to be efficient
        with the remaining budget.
      - Transition is smooth via a sigmoid on total cost.

    mu, sigma, cost: shape (2000,)
    f_best: scalar, lowest observed value so far.
    Returns: shape (2000,) finite scores, higher = more desirable.
    """
    # Deadline guard: if we're running out of time, fall back to a simple
    # cost-aware EI that is fast to compute.
    if _time_remaining() < 0.5:
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)

    # Total cost spent so far (sum of cost array gives total for 2000 candidates,
    # but cost[i] is the cost of evaluating candidate i. The total budget spent
    # is tracked externally; here we use the mean cost as a proxy for phase).
    # Actually, 'cost' has shape (2000,) and represents the cost of each candidate.
    # The total cost spent so far is not directly available, but we can infer
    # the phase from the magnitude of f_best relative to the known minimum
    # (-3.86278) or from the spread of sigma.
    #
    # Better approach: use the mean cost as a proxy. In early iterations,
    # the GP is uncertain (high sigma) and f_best is far from optimum.
    # We'll use a combination:
    #   - If sigma is large on average, we're in exploration phase.
    #   - If sigma is small, we're in exploitation phase.

    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma)

    # Normalize sigma to get a "certainty" measure.
    # When sigma is large relative to the range, we're uncertain -> explore.
    # Hartmann 3d range is roughly [-3.86, 0], so scale ~ 4.
    sigma_norm = sigma_mean / (4.0 + 1e-9)

    # Transition: when sigma_norm is high (uncertain), weight cost less.
    # When sigma_norm is low (certain), weight cost more.
    # Use a smooth transition: weight = 1 / (1 + alpha * sigma_norm)
    # where alpha controls the sharpness.
    alpha = 3.0
    cost_weight = 1.0 / (1.0 + alpha * sigma_norm)

    # Compute EI
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Ensure non-negative EI (clip small negatives from numerical error)
    ei = np.maximum(ei, 0.0)

    # Cost-aware score: blend between pure EI and cost-aware EI
    # score = cost_weight * EI + (1 - cost_weight) * EI / cost
    # This is a smooth interpolation.
    cost_safe = np.maximum(cost, 1e-9)
    ei_cost_aware = ei / cost_safe

    score = cost_weight * ei + (1.0 - cost_weight) * ei_cost_aware

    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
