"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm


def _safe_ei(mu, sigma, f_best):
    """Compute expected improvement with numerical safeguards."""
    sigma = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma
    # Clip z to avoid overflow in cdf/pdf
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    return np.maximum(ei, 0.0)


def _safe_ucb(mu, sigma, f_best, xi):
    """Compute UCB with numerical safeguards."""
    return mu - xi * sigma


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function: blends cost-aware EI with exploration via UCB.
    
    Parameters:
    - mu: array of shape (2000,), GP mean predictions
    - sigma: array of shape (2000,), GP std dev predictions
    - f_best: float, best observed value so far
    - cost: array of shape (2000,), evaluation costs (1 + mean(x_scaled))
    
    Returns:
    - scores: array of shape (2000,), higher = evaluate next
    """
    # Deadline guard: if we've been running too long, return a simple heuristic
    # But since this is called per-iteration and each call should be fast,
    # we mainly need to ensure we don't do anything expensive.
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Ensure inputs are 1D numpy arrays
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    cost = np.asarray(cost, dtype=np.float64).ravel()
    
    # Safety: replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Compute cost-aware EI
    ei = _safe_ei(mu, sigma, f_best)
    cost_ei = ei / cost
    
    # Compute UCB for exploration (xi is a tunable parameter)
    # Use a moderate xi that balances exploitation and exploration
    xi = 2.0
    ucb = _safe_ucb(mu, sigma, f_best, xi)
    
    # Normalize UCB to be comparable with cost_ei
    # UCB is in objective units (lower is better), so we negate to make higher = better
    ucb_score = -ucb
    
    # Blend: primarily use cost-aware EI, but add a small UCB component
    # to encourage exploration in uncertain regions
    # The UCB term is scaled down so it doesn't dominate
    ucb_weight = 0.1
    
    # Normalize both to [0, 1] range for blending
    ei_min, ei_max = np.min(cost_ei), np.max(cost_ei)
    ucb_min, ucb_max = np.min(ucb_score), np.max(ucb_score)
    
    if ei_max > ei_min:
        ei_norm = (cost_ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.ones_like(cost_ei)
    
    if ucb_max > ucb_min:
        ucb_norm = (ucb_score - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.ones_like(ucb_score)
    
    # Blended score
    scores = (1 - ucb_weight) * ei_norm + ucb_weight * ucb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
