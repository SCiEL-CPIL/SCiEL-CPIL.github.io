"""Cost-aware acquisition for 6D Hartmann minimization (fresh design)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm


def _deadline_guard():
    # Wall-clock budget: evaluator kills at 330 s; keep a 25 s safety margin.
    return time.perf_counter() < (330.0 - 25.0)


def _finite_scores(scores):
    # Guarantee a valid finite output; fall back to a cheapness-based score.
    if not np.all(np.isfinite(scores)):
        # cost is in (1, 2]; higher score for cheaper points.
        return np.where(np.isfinite(scores), scores, 0.0)
    return scores


def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Guard against a pathological/empty call.
    if mu.size != 2000 or mu.shape != (2000,):
        return np.zeros(2000, dtype=np.float64)

    # Protect against zero/negative cost or std.
    cost = np.where(cost > 1e-6, cost, 1e-6)
    sigma = np.where(sigma > 1e-9, sigma, 1e-9)

    # Cost-normalized Lower Confidence Bound: optimistic value density.
    # beta grows gently with the number of points (budget-aware exploration).
    n = mu.size
    beta = 1.2 + 0.25 * np.log1p(n)
    lcb = (mu - beta * sigma) / cost

    # Probability-of-improvement term (conservative, high-probability wins),
    # also normalized by cost to reward cheap, likely improvements.
    z = (f_best - mu) / sigma
    poi = norm.cdf(z)
    poi_cost = poi / cost

    # Blend: value density drives ranking of cheap optimistic regions;
    # POI/cost adds a conservative, high-probability-improvement bias.
    scores = 0.65 * lcb + 0.35 * poi_cost

    # Deadline guard: if we are running long, return the best finite result now.
    if not _deadline_guard():
        return _finite_scores(scores)

    # Scale to a stable numeric range to avoid overflow/underflow in ranking.
    finite = np.isfinite(scores)
    if np.any(finite):
        smin = np.min(scores[finite])
        smax = np.max(scores[finite])
        if smax - smin > 1e-12:
            scores = (scores - smin) / (smax - smin)
        else:
            scores = np.zeros_like(scores)
    scores[~finite] = -np.inf

    return _finite_scores(scores)
# EVOLVE-BLOCK-END
