"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE_SEC = 300.0  # explicit safety margin before the 330 s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower-Confidence-Bound (LCB) acquisition for minimization.

    Structural change from the seed EI/cost: instead of a risk-sensitive
    expected-improvement score we use an optimism-in-the-face-of-uncertainty
    lower bound. For a minimization problem the optimistic (lower) prediction is
        lcb(x) = mu(x) - beta * sigma(x)        (lower is better)
    and we return a score where HIGHER means "evaluate next":
        score(x) = (-mu(x) + beta * sigma(x)) / cost(x)

    The exploration weight beta is data-driven (it scales with the GP's global
    predictive uncertainty relative to the objective scale) rather than a fixed
    hand-tuned constant, so exploration is strong while the model is uncertain
    (early) and tightens as the posterior concentrates (late).

    Args:
        mu (np.ndarray): GP mean predictions, shape (N,)
        sigma (np.ndarray): GP std predictions, shape (N,)
        f_best (float): best (lowest) observed value so far
        cost (np.ndarray): evaluation cost per candidate, shape (N,)

    Returns:
        np.ndarray: acquisition scores, shape (N,), finite, higher = evaluate next
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Guard against zero/negative sigma.
    sigma_safe = np.where(sigma > 1e-10, sigma, 1e-10)

    # Data-driven exploration weight.
    # Reference objective scale from the best value seen so far.
    ref_scale = max(abs(f_best), 1e-8)
    # Global predictive uncertainty (median is robust to a few hot spots).
    sigma_global = float(np.median(sigma_safe))
    # beta grows with relative uncertainty; clamp to a sane band.
    beta = 1.0 + 2.0 * (sigma_global / ref_scale)
    beta = float(np.clip(beta, 0.5, 4.0))

    # Cost-aware LCB score (higher = better).
    cost_safe = np.maximum(cost, 1e-10)
    score = (-mu + beta * sigma_safe) / cost_safe

    # Guarantee finiteness: replace any non-finite entry with a large finite
    # negative value (a -inf return would invalidate the whole result).
    score = np.where(np.isfinite(score), score, -1e30)

    # Explicit deadline guard: the computation is vectorized and tiny, but if we
    # somehow approach the wall-clock limit we still return a valid finite array.
    if (time.time() - _START) > _DEADLINE_SEC:
        return score

    return score
# EVOLVE-BLOCK-END
