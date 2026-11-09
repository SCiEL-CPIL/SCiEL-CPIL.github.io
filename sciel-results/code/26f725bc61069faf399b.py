"""Cost-aware acquisition function for Bayesian optimization (4 test functions)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_T0 = time.perf_counter()
_DEADLINE = 330.0 - 25.0  # safety margin before the 330 s wall-clock kill


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition: UCB per unit cost with adaptive exploration.

    mu, sigma, cost: 1-D arrays of shape (N,). Returns N finite scores.
    """
    t0 = time.perf_counter()
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Guard against non-finite / non-positive inputs (defensive; should not trigger).
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    sigma = np.maximum(sigma, 0.0)

    # Numerically safe computation
    s = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-6)

    # Compute Expected Improvement (f_best - mu) is the improvement for minimization.
    z = (f_best - mu) / s
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Compute UCB for minimization: potential improvement + exploration bonus
    # beta adapts based on current uncertainty level relative to improvement
    # Higher beta when sigma is large relative to (f_best - mu)
    improvement = f_best - mu
    # Adaptive beta: scale with uncertainty relative to improvement magnitude
    # Use a soft cap to avoid extreme values
    ratio = s / (np.abs(improvement) + 1e-9)
    beta = 1.0 + 1.0 * np.tanh(ratio)  # beta in [1, 2]

    ucb = improvement + beta * s

    # Combine EI and UCB: weighted average, emphasizing UCB when uncertainty is high
    # w = 0.5 * tanh(sigma / (|f_best - mu| + 1e-9)) gives weight to UCB
    w = 0.5 * np.tanh(s / (np.abs(improvement) + 1e-9))
    combined = (1 - w) * ei + w * np.maximum(ucb, 0.0)

    # Cost-aware ranking: improvement per unit cost, plus a tiny exploration term
    score = combined / cost_safe + 1e-4 * s / cost_safe

    # Ensure all outputs are finite.
    score = np.where(np.isfinite(score), score, 0.0)

    # Deadline guard: return the best (finite) result found before the limit.
    if (time.perf_counter() - t0) > (_DEADLINE - (time.perf_counter() - _T0)):
        return score
    return score
# EVOLVE-BLOCK-END
