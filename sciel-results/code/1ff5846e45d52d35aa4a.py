"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track total elapsed time for deadline guard
_T0 = time.time()
_DEADLINE = 315.0  # 300s limit + 15s safety margin


def _time_left():
    return _DEADLINE - (time.time() - _T0)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: EI/cost with a small sigma-based exploration bonus.
    
    Uses the proven cost-normalized Expected Improvement as the core signal,
    augmented by a gentle 5%-capped uncertainty bonus to encourage exploration
    of unexplored regions without distorting exploitation rankings.
    
    Parameters
    ----------
    mu : np.ndarray, shape (N,)
        GP mean predictions.
    sigma : np.ndarray, shape (N,)
        GP standard deviation predictions.
    f_best : float
        Best observed objective value (lower is better).
    cost : np.ndarray, shape (N,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    np.ndarray, shape (N,)
        Acquisition scores (higher = evaluate next).
    """
    # Deadline guard: if we've been running too long, return a simple, fast score
    if _time_left() < 5.0:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return ei / cost_safe

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma and cost to small positive values
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # --- Core: Cost-normalized Expected Improvement ---
    # Standard EI for minimization: (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10.0, 10.0)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Primary signal: EI per unit cost
    ei_cost = ei / cost_safe
    
    # --- Small exploration bonus ---
    # Normalize sigma to [0, 1] and add a capped bonus (max 5% of max EI/cost)
    # This gently encourages unexplored regions without overwhelming exploitation
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    sigma_range = sigma_max - sigma_min
    
    if sigma_range > 1e-12:
        sigma_norm = (sigma_safe - sigma_min) / sigma_range
    else:
        sigma_norm = np.ones_like(sigma_safe) * 0.5
    
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
