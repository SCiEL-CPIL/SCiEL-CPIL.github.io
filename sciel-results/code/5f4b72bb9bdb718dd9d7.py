# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 300.0


def _check_deadline():
    return (time.time() - _START_TIME) > (_DEADLINE - 5.0)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: EI/cost with a small sigma-based exploration bonus.
    """
    if _check_deadline():
        idx = np.argmin(cost)
        scores = np.full_like(mu, -1e10)
        scores[idx] = 1.0
        return scores

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Primary signal: cost-normalized EI (the proven 1.0-scoring approach)
    ei_cost = ei / cost_safe

    # Small exploration bonus: reward candidates in high-uncertainty regions.
    # Scale by max EI to keep it a gentle nudge, not a dominant term.
    max_ei = np.max(ei_cost)
    if max_ei > 1e-12:
        # Normalize sigma to [0,1] range for stable bonus
        sigma_min = np.min(sigma_safe)
        sigma_max = np.max(sigma_safe)
        sigma_range = sigma_max - sigma_min
        if sigma_range > 1e-12:
            sigma_norm = (sigma_safe - sigma_min) / sigma_range
        else:
            sigma_norm = np.full_like(sigma_safe, 0.5)
        # Small weight: 5% of max EI at most
        exploration_bonus = 0.05 * max_ei * sigma_norm
    else:
        exploration_bonus = np.zeros_like(ei_cost)

    scores = ei_cost + exploration_bonus
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
# EVOLVE-BLOCK-END
