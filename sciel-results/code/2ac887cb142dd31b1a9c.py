"""Cost-aware acquisition for Bayesian optimization with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # 10s safety margin before 300s limit

def _time_remaining():
    """Return seconds remaining before hard deadline."""
    return _DEADLINE - time.time()


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining EI and UCB with adaptive blending.
    
    Uses a model-and-optimize approach:
    1. Compute cost-normalized EI (expected improvement per unit cost)
    2. Compute cost-normalized UCB (uncertainty per unit cost)
    3. Adaptively blend based on optimization progress (sigma relative to f_best)
    
    The blend shifts from UCB-dominated (early, high uncertainty) to
    EI-dominated (late, low uncertainty) as optimization progresses.
    """
    # Time guard: if we're running low, return a simple safe answer
    if _time_remaining() < 3.0:
        mu_arr = np.asarray(mu, dtype=np.float64)
        sigma_safe = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-6)
        cost_arr = np.maximum(np.asarray(cost, dtype=np.float64), 1.0)
        return (-mu_arr + 2.0 * sigma_safe) / cost_arr

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Safe versions
    sigma_safe = np.maximum(sigma, 1e-6)
    cost_safe = np.maximum(cost, 1.0)  # cost is at least 1

    # === Adaptive parameter scheduling ===
    # Key insight: as f_best decreases (improves), shift from exploration to exploitation
    sigma_mean = np.mean(sigma_safe)
    
    if abs(f_best) > 1e-6:
        sigma_rel = sigma_mean / abs(f_best)
    else:
        sigma_rel = 1.0
    
    # Clamp to reasonable range
    sigma_rel = np.clip(sigma_rel, 0.0, 5.0)
    
    # UCB exploration coefficient: from 3.0 (early) to 1.0 (late)
    kappa = 3.0 - 2.0 * np.clip(sigma_rel / 3.0, 0.0, 1.0)
    
    # EI weight: from 0.3 (early, more UCB) to 0.8 (late, more EI)
    w_ei = 0.3 + 0.5 * np.clip(1.0 - sigma_rel / 3.0, 0.0, 1.0)
    w_ucb = 1.0 - w_ei
    
    # Cost penalty exponent: from 0.3 (early, light penalty) to 1.0 (late, full penalty)
    alpha = 0.3 + 0.7 * np.clip(1.0 - sigma_rel / 3.0, 0.0, 1.0)

    # === Compute Expected Improvement (EI) ===
    # For minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    # Numerical stability: clip z to avoid overflow
    z = np.clip(z, -10.0, 10.0)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # === Compute UCB (for minimization, maximize -mu + kappa*sigma) ===
    ucb = -mu + kappa * sigma_safe

    # === Cost-aware adjustment ===
    # Cost-normalized EI: EI / cost^alpha
    ei_cost = ei / (cost_safe ** alpha)
    
    # Cost-normalized UCB: UCB / cost
    ucb_cost = ucb / cost_safe

    # === Normalize both to comparable scales ===
    ei_c_min = np.min(ei_cost)
    ei_c_max = np.max(ei_cost)
    ei_c_range = ei_c_max - ei_c_min
    if ei_c_range > 1e-12:
        ei_c_norm = (ei_cost - ei_c_min) / ei_c_range
    else:
        ei_c_norm = np.zeros_like(ei_cost)

    ucb_min = np.min(ucb_cost)
    ucb_max = np.max(ucb_cost)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-12:
        ucb_norm = (ucb_cost - ucb_min) / ucb_range
    else:
        ucb_norm = np.zeros_like(ucb_cost)

    # === Weighted combination ===
    score = w_ei * ei_c_norm + w_ucb * ucb_norm

    # Ensure finite and non-negative
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)

    return score
# EVOLVE-BLOCK-END
