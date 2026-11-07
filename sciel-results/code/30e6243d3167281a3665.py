"""Cost-aware acquisition for BO on three hump camel 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_TIME_LIMIT = 320.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a UCB-on-reward formulation.
    
    Strategy:
    1. Compute Expected Improvement (EI) for minimization.
    2. Compute a cost-aware UCB on the reward: score = (-mu + beta * sigma) / sqrt(cost).
       - beta is scaled by remaining budget ratio to shift from exploration to exploitation.
    3. Blend EI and UCB scores with an adaptive weight that increases EI's influence
       as f_best approaches the minimum (0).
    
    Parameters:
        mu: (2000,) GP mean predictions in objective units
        sigma: (2000,) GP std dev predictions (positive) in objective units
        f_best: float, best observed value so far
        cost: (2000,) evaluation cost per candidate
    
    Returns:
        (2000,) finite acquisition scores, higher = better
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        return np.zeros(len(mu), dtype=np.float64)
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1.0)
    
    # --- Expected Improvement (minimization) ---
    Z = (f_best - mu) / sigma_safe
    Z = np.clip(Z, -10.0, 10.0)
    phi = norm.pdf(Z)
    Phi = norm.cdf(Z)
    EI = (f_best - mu) * Phi + sigma_safe * phi
    EI = np.maximum(EI, 0.0)
    
    # --- Cost-aware UCB on reward ---
    # Reward for candidate x is -f(x) (higher is better).
    # UCB on reward: -mu + beta * sigma
    # Normalize by sqrt(cost) for cost-awareness.
    
    # Adaptive beta:
    # Remaining budget ratio: we don't know total cost spent, but we can use
    # f_best as a proxy for progress. Early on f_best is large -> explore more.
    # f_scale ~ 5.5 (max of three-hump camel 2D on [-5,5]^2)
    f_scale = 5.5
    progress = np.clip(1.0 - f_best / f_scale, 0.0, 1.0)
    
    # beta starts high (exploration) and decays as we make progress.
    # At progress=0: beta ~ 2.0 (strong exploration)
    # At progress=1: beta ~ 0.5 (exploitation)
    beta = 2.0 - 1.5 * progress
    
    # UCB reward: -mu + beta * sigma
    ucb_reward = -mu + beta * sigma_safe
    
    # Cost-normalized UCB
    cost_norm = np.sqrt(cost_safe)
    ucb_score = ucb_reward / cost_norm
    
    # --- Adaptive blend of EI and UCB ---
    # Early in search (low progress): rely more on UCB for exploration
    # Late in search (high progress): rely more on EI for exploitation
    
    # Normalize both scores to comparable scale
    ei_max = np.max(EI) if n > 0 else 1.0
    ucb_max = np.max(ucb_score) if n > 0 else 1.0
    ei_min = np.min(EI) if n > 0 else 0.0
    ucb_min = np.min(ucb_score) if n > 0 else 0.0
    
    ei_range = ei_max - ei_min if ei_max > ei_min else 1.0
    ucb_range = ucb_max - ucb_min if ucb_max > ucb_min else 1.0
    
    ei_norm = (EI - ei_min) / ei_range
    ucb_norm = (ucb_score - ucb_min) / ucb_range
    
    # Weight: w_ei increases with progress
    # At progress=0: w_ei = 0.3 (mostly UCB)
    # At progress=1: w_ei = 0.8 (mostly EI)
    w_ei = 0.3 + 0.5 * progress
    w_ucb = 1.0 - w_ei
    
    score = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Scale back to a reasonable magnitude
    score = score * 10.0
    
    # Add a small exploration bonus for high-sigma, low-EI regions
    ei_threshold = np.percentile(EI, 25) if n > 0 else 0
    low_ei_mask = EI < ei_threshold
    sigma_max = np.max(sigma_safe) if np.max(sigma_safe) > 0 else 1.0
    bonus = 0.05 * (sigma_safe / sigma_max) / np.sqrt(cost_safe)
    score[low_ei_mask] += bonus[low_ei_mask]
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
