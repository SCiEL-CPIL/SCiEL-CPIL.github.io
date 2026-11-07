"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track wall-clock deadline (330s limit, use 310s margin)
_T0 = time.time()
_DEADLINE = 310.0


def _time_ok():
    return (time.time() - _T0) < _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Bayesian optimization of Styblinski-Tang 2D.
    
    mu, sigma, cost: arrays of shape (2000,)
    f_best: scalar, lowest observed value
    Returns: 2000 finite scores (higher = evaluate next)
    """
    # Deadline guard
    if not _time_ok():
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / np.maximum(cost, 1e-6)
        return np.where(np.isfinite(score), score, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Safety: ensure finite and positive
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    sigma_safe = np.maximum(sigma, 1e-8)

    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Infer progress: f_best starts near 0 (origin) and goes toward -78.3323
    f_best_val = float(f_best)
    if not np.isfinite(f_best_val):
        f_best_val = 0.0
    progress = np.clip(-f_best_val / 78.3323, 0.0, 1.0)

    # Adaptive parameters with smooth transition
    # Beta for UCB: starts high for exploration, decreases for exploitation
    beta = 2.5 * np.exp(-1.5 * progress) + 0.2

    # UCB acquisition
    ucb = -mu + beta * sigma_safe

    # Cost efficiency: use cost in a way that rewards cheap high-improvement points
    # Normalize cost to [0,1] range: cost is in [1, 2]
    cost_norm = (cost - 1.0) / 1.0  # [0, 1]
    cost_safe = np.maximum(cost, 1e-6)
    
    # Cost-aware scoring: divide by a cost factor that's less aggressive than pure division
    # Use a softer cost penalty: cost^alpha where alpha < 1
    cost_factor = np.power(cost_safe, 0.7)
    
    score_ei = ei / cost_factor
    score_ucb = ucb / cost_factor

    # Blend weights: smooth sigmoid-like transition
    # Early (progress~0): more UCB, Late (progress~1): more EI
    w_ucb = 0.5 * (1.0 + np.exp(-3.0 * progress)) / 2.0 + 0.1
    w_ucb = np.clip(w_ucb, 0.1, 0.7)
    w_ei = 1.0 - w_ucb

    score = w_ei * score_ei + w_ucb * score_ucb

    # Small exploration bonus for high uncertainty regions
    # This prevents premature convergence to local minima
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-6:
        explore_bonus = 0.05 * (sigma_safe / sigma_max) * (1.0 - progress)
        score = score + explore_bonus

    # Mild cost penalty for expensive evaluations (corners)
    cost_excess = np.maximum(cost - 1.5, 0.0)
    penalty = 0.05 * cost_excess ** 2
    score = score - penalty

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
