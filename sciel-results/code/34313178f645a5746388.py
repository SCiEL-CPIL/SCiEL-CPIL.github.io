"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_T0 = time.time()
_DEADLINE = _T0 + 320.0  # 10s margin before the 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Returns scores (higher = evaluate next) for 2000 candidate points.
    Combines expected improvement per unit cost with a diversity bonus
    to encourage exploration.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Safety: ensure finite inputs
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma)) and np.all(np.isfinite(cost))):
        return np.zeros_like(mu)

    # Deadline guard: if we're close to the limit, return a simple exploitation score
    if time.time() > _DEADLINE:
        # Return low-mu (exploitation) normalized by cost
        return -mu / (cost + 1e-9)

    # Standardize sigma for numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)

    # Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Cost-aware EI: divide by cost
    cost_safe = np.maximum(cost, 1e-9)
    ei_cost = ei / cost_safe

    # Add a small diversity/exploration term based on sigma
    # Normalize sigma to [0, 1] range for the bonus
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    if sigma_max > sigma_min:
        sigma_norm = (sigma_safe - sigma_min) / (sigma_max - sigma_min)
    else:
        sigma_norm = np.zeros_like(sigma_safe)

    # Combine: 90% cost-aware EI + 10% exploration bonus
    score = 0.9 * ei_cost + 0.1 * sigma_norm

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
