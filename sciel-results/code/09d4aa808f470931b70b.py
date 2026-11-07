"""Cost-aware acquisition for Bayesian optimization of powell 4d (min 0) on [-4,5]^4."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Safety margin before the 330 s wall-clock kill.
_DEADLINE_S = 330.0
_SAFETY_S = 2.0
_EPS = 1e-9


def acquisition(mu, sigma, f_best, cost):
    t0 = time.time()
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Guard against non-finite inputs.
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, _EPS)
    sigma = np.clip(sigma, _EPS, None)

    # Ensure cost is valid (positive and finite).
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.5)
    cost = np.clip(cost, _EPS, None)

    # Expected Improvement relative to the current best observed value.
    # improvement = f_best - mu (positive if mu < f_best, i.e., potential improvement)
    improvement = f_best - mu
    z = improvement / sigma

    # Expected improvement (standard form).
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)

    # Cost-aware score: EI / cost
    # This ensures that points with lower expected improvement per unit cost are preferred.
    # Higher EI and lower cost lead to higher scores.
    score = ei / cost

    # Ensure finiteness and a higher-is-better orientation.
    score = np.where(np.isfinite(score), score, 0.0)

    # Deadline guard: if we are about to exceed the budget, return the best available.
    if time.time() - t0 > (_DEADLINE_S - _SAFETY_S):
        return np.where(np.isfinite(score), score, 0.0)

    return np.where(np.isfinite(score), score, 0.0)
# EVOLVE-BLOCK-END
