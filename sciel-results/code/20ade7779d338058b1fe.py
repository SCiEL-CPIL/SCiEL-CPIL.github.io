import numpy as np
from scipy.stats import norm
import time

# Global timer for safety (wall-clock, matches the 330s evaluator kill)
_START_TIME = time.time()
_TIME_LIMIT = 320.0  # Safety margin below the 330s kill


def _fast_fallback(mu, sigma, f_best, cost):
    """Cheap Cost-aware LCB used only if we are near the deadline."""
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    beta = 0.5 * float(np.max(sigma_safe))
    lcb = mu - beta * sigma_safe
    scores = lcb / np.power(cost_safe, 0.7)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Bayesian optimization (minimize cosine8 8d).

    Structural choice: Adaptive Cost-aware LCB (Upper Confidence Bound for
    minimization) with an exploration coefficient scaled by the maximum sigma
    in the candidate set, plus a small Expected-Improvement exploitation term.
    Scores are divided by a (sub-linear) cost factor and given a cheap-region
    bonus, so the loop prefers high-value, low-cost evaluations.

    mu, sigma, cost : shape (2000,)
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Explicit deadline guard: bail to the fast path if we are running out.
    if time.time() - _START_TIME > _TIME_LIMIT:
        return _fast_fallback(mu, sigma, f_best, cost)

    # Safety in denominators / magnitudes
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Expected Improvement (minimization) — exploitation signal
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Adaptive exploration coefficient in objective units.
    # Scale by the max sigma so the term shrinks as the GP tightens,
    # but remains meaningful while uncertainty is high.
    max_sigma = float(np.max(sigma_safe))
    beta = 0.5 * max_sigma

    # LCB for minimization (lower predicted value with an uncertainty bonus)
    lcb = mu - beta * sigma_safe

    # Combine exploration (LCB) with a light exploitation (EI) term.
    # EI is in objective units and already small where the GP is confident.
    value = lcb + 0.1 * ei

    # Cost-aware: penalize expensive evaluations sub-linearly (so we still
    # explore moderately-costly regions) and reward cheap ones.
    cost_factor = np.power(cost_safe, 0.7)
    scores = value / cost_factor

    # Cheap-region bonus: cost in [1, 2]; cheaper (closer to 1) is better.
    # (2 - cost) is in [0, 1]; weight it modestly relative to the value scale.
    cheap_bonus = 0.05 * (2.0 - cost_safe) * max_sigma
    scores = scores + cheap_bonus

    # Ensure all outputs are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
