"""
Strategy: Cost-aware Expected Improvement with adaptive exploration via LCB blend.

Key structural changes from the seed:
1. Use a principled blend of EI (exploit) and LCB (explore) with an adaptive
   exploration coefficient that responds to the global uncertainty landscape.
2. Apply a soft cost penalty (cost ** 0.5) to normalize scores by evaluation cost.
3. Guard against numerical issues and enforce a wall-clock deadline.
"""

# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time

    if _start_time is None:
        _start_time = time.time()

    elapsed = time.time() - _start_time

    # --- Numerical safety ---
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    cost_safe = np.where(cost > 1e-9, cost, 1e-9)

    # --- Global uncertainty statistics for adaptive beta ---
    sigma_95 = np.percentile(sigma_safe, 95)
    sigma_max = np.max(sigma_safe)

    # Adaptive beta: scale exploration with the spread of uncertainty relative
    # to the current best value. When f_best is close to 0 (good) and sigma is
    # small, beta shrinks (exploit). When sigma is large or f_best is far from 0,
    # beta grows (explore).
    denom = abs(f_best) + sigma_95 + 1e-9
    beta = 2.5 * (sigma_95 / denom)
    beta = np.clip(beta, 0.2, 4.0)

    # --- Expected Improvement (EI) for minimization ---
    z = (f_best - mu) / sigma_safe
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = (f_best - mu) * cdf + sigma_safe * pdf

    # --- Lower Confidence Bound (LCB) for minimization ---
    lcb = -mu + beta * sigma_safe

    # --- Blend EI and LCB ---
    # Weight EI more heavily for exploitation, LCB for exploration.
    w_ei = 0.6
    w_lcb = 0.4
    blended = w_ei * ei + w_lcb * lcb

    # --- Cost penalty ---
    # Softer penalty to avoid over-penalizing expensive but promising points.
    cost_penalty = np.power(cost_safe, 0.5)

    scores = blended / cost_penalty

    # --- Ensure finite values ---
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # --- Deadline guard ---
    if elapsed > 290.0:
        # Fallback to a very fast score if we are close to the limit.
        return ei / cost_safe

    return scores
# EVOLVE-BLOCK-END
