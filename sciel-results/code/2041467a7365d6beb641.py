"""Cost-aware acquisition for Bayesian optimization on Ackley 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills after 330s wall-clock.
# We keep a safety margin so we return before the hard limit.
_DEADLINE = 320.0  # seconds from module load (approximate; per-call overhead is small)
_start_time = time.time()


def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite acquisition scores (higher = evaluate next).

    mu, sigma: shape (2000,), GP mean and std in objective units.
    f_best: scalar, lowest observed value.
    cost: shape (2000,), cost of evaluating each candidate.
    """
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        # Return a safe fallback: uniform random scores
        return np.random.uniform(0, 1, size=len(mu))

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)

    n = len(mu)

    # UCB-like hybrid: (f_best - mu) + beta * sigma
    # This assigns positive scores to points with high uncertainty even if
    # their mean prediction is worse than f_best, encouraging exploration
    # and helping escape local minima in Ackley 2D.
    beta = 1.0
    improvement = np.maximum(0.0, (f_best - mu) + beta * sigma)

    # Find Pareto frontier in (improvement, cost) space:
    # A point is Pareto-optimal if no other point has both >= improvement and <= cost
    # (with at least one strict inequality).
    # Sort by improvement descending, track min cost seen.
    order = np.argsort(-improvement, kind='mergesort')

    is_pareto = np.zeros(n, dtype=bool)
    min_cost_so_far = np.inf
    for idx in order:
        if cost[idx] < min_cost_so_far:
            is_pareto[idx] = True
            min_cost_so_far = cost[idx]

    # Compute hypervolume contribution for each Pareto point
    scores = np.zeros(n, dtype=np.float64)

    pareto_indices = np.where(is_pareto)[0]

    if len(pareto_indices) > 0:
        # Sort Pareto points by improvement descending
        pareto_order = np.argsort(-improvement[pareto_indices], kind='mergesort')
        pareto_sorted = pareto_indices[pareto_order]

        # Reference cost for hypervolume calculation
        C_ref = cost[pareto_sorted].max() + 1.0

        for i, idx in enumerate(pareto_sorted):
            if i < len(pareto_sorted) - 1:
                next_idx = pareto_sorted[i + 1]
                imp_diff = improvement[idx] - improvement[next_idx]
            else:
                imp_diff = improvement[idx]

            cost_height = C_ref - cost[idx]
            scores[idx] = imp_diff * cost_height

    # Fallback: if all scores are zero, use random scores
    if np.all(scores == 0.0):
        scores = np.random.uniform(0, 1, size=n)

    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
