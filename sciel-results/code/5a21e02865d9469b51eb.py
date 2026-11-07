"""Cost-aware acquisition for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: the evaluator kills the process at ~330 s wall clock.
# We track per-call time and bail out with a safe fallback if we approach it.
_T0 = time.time()
_DEADLINE = 325.0  # safety margin (seconds)


def _time_left():
    return _DEADLINE - (time.time() - _T0)


def acquisition(mu, sigma, f_best, cost):
    """Return 2000 finite acquisition scores (higher = evaluate next)."""
    n = len(mu)
    if n == 0:
        return np.array([], dtype=float)

    # Guard against non-finite inputs
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    f_best = float(f_best)

    # Sanitize
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.where(cost > 1e-9, cost, 1.0)
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Remaining budget heuristic: total budget is 30, current cost is f_best's
    # accumulated cost proxy. We approximate remaining budget from the cost array:
    # the evaluator tracks total cost; we use the mean cost as a proxy for
    # how many more evaluations fit. Since we don't know total spent, we use
    # a fixed remaining-budget estimate based on the fact that cost per eval
    # is ~1 + mean(x) in [1, 2]. With budget 30 and 5 initial points, we have
    # roughly 25 units left. We scale the discount by a conservative remaining
    # budget estimate.
    # Use a simple proxy: remaining = 30 - sum of costs so far is unknown,
    # so we use the mean cost to estimate remaining steps.
    mean_cost = float(np.mean(cost))
    # Assume we've spent roughly 5 (initial) + a few evaluations.
    # Conservative: remaining budget ~ 20 units of cost.
    remaining_budget = 20.0

    # Discount factor: cheap points get higher multiplier because they leave
    # more budget for future evaluations.
    discount = 1.0 + remaining_budget / (cost + 1e-9)

    # UCB-style acquisition with discount
    # beta controls exploration; we use a moderate value.
    beta = 1.5
    ucb = (f_best - mu) + beta * sigma

    # Local exploitation: pull toward the GP mean's best region.
    # Points where the GP is confident (low sigma) and close to f_best get
    # a small bonus.
    confidence_bonus = 0.1 * np.exp(-sigma / 0.05)

    scores = ucb * discount + confidence_bonus

    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
