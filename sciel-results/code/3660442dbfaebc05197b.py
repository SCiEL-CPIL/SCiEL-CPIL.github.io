"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Module-level deadline guard: 330s wall-clock limit, use 320s as safety margin
_T0 = time.time()
_DEADLINE = _T0 + 320.0


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're running out of time, return simple EI/cost
    # without any extra computation
    if time.time() > _DEADLINE:
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        score = ei / np.maximum(cost, 1e-8)
        score = np.where(np.isfinite(score), score, 0.0)
        return score

    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)

    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Cost-aware base score: EI / cost
    base_score = ei / cost_safe

    # Add a small exploration bonus for high uncertainty, scaled by cost
    # This helps escape local optima in the 6D space while still
    # preferring cheaper evaluations
    exploration_bonus = 0.05 * sigma_safe / cost_safe

    score = base_score + exploration_bonus

    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
