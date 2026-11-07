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
    Cost-aware acquisition function for Bayesian optimization on Levy 2D.
    
    Strategy: Cost-normalized Expected Improvement with a small capped
    exploration bonus based on GP uncertainty (sigma). This reverts to the
    proven EI/cost core that scored ~1.0 in multiple runs, while adding a
    gentle 5%-capped sigma-based bonus to encourage exploration of
    unexplored regions without distorting the exploitation ranking.
    """
    if _check_deadline():
        # Return a safe, deterministic fallback if we're near the deadline
        idx = np.argmin(cost)
        scores = np.full_like(mu, -1e10)
        scores[idx] = 1.0
        return scores

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = len(mu)

    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)

    # --- Core: Cost-normalized Expected Improvement ---
    # Standard EI: (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    ei_cost = ei / cost_safe

    # --- Small exploration bonus ---
    # Normalize sigma to [0, 1] and add a capped bonus (max 5% of max EI/cost)
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    sigma_range = sigma_max - sigma_min
    if sigma_range > 1e-12:
        sigma_norm = (sigma_safe - sigma_min) / sigma_range
    else:
        sigma_norm = np.ones_like(sigma_safe) * 0.5

    # Cap the bonus at 5% of the maximum EI/cost value
    ei_cost_max = np.max(ei_cost)
    if ei_cost_max > 1e-12:
        bonus_cap = 0.05 * ei_cost_max
    else:
        bonus_cap = 0.0

    exploration_bonus = bonus_cap * sigma_norm

    # Final score: EI/cost + small exploration bonus
    scores = ei_cost + exploration_bonus

    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
