"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track elapsed time to avoid timeout
_START_TIME = time.time()
_DEADLINE_SECONDS = 325.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using LCB with adaptive exploration.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed objective value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x scaled to [0,1])).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = better candidate to evaluate next).
    """
    # Deadline check
    if time.time() - _START_TIME > _DEADLINE_SECONDS:
        # Return a safe fallback: prioritize low cost and low mu
        return -(mu + 0.5 * sigma) / (cost + 1e-9)
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clamp sigma to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Normalize cost to [0, 1] range for consistent weighting
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range < 1e-12:
        cost_norm = np.zeros_like(cost)
    else:
        cost_norm = (cost - cost_min) / cost_range
    
    # Adaptive exploration coefficient: decays as we spend more budget
    # f_best is in objective units; as f_best approaches 0 (the minimum),
    # we should exploit more
    # Use a sigmoid-like decay based on how close f_best is to 0
    # Levy 2d has minimum 0, so f_best >= 0
    # When f_best is large (early), explore more; when small, exploit more
    f_best_abs = abs(f_best)
    # Scale: at f_best=10, beta~2.0 (explore); at f_best=0.1, beta~0.3 (exploit)
    beta = 0.3 + 1.7 * np.exp(-f_best_abs / 2.0)
    
    # LCB: lower confidence bound (we want to minimize, so lower mu + beta*sigma is better)
    # Acquisition score should be higher for better candidates
    # LCB value = mu - beta * sigma (lower is better for minimization)
    # So score = -LCB = -mu + beta * sigma
    lcb_score = -mu + beta * sigma_safe
    
    # Cost penalty: penalize high-cost evaluations
    # Use a multiplicative factor: divide by cost to prefer cheap points
    # But also add a small exploration bonus for uncertain, cheap points
    cost_penalty = 1.0 / (cost + 1e-9)
    
    # Combine: weighted combination of LCB and cost-awareness
    # As f_best improves, weight LCB more heavily
    lcb_weight = 0.7 + 0.3 * np.exp(-f_best_abs / 5.0)
    cost_weight = 1.0 - lcb_weight
    
    # Normalize lcb_score to have zero mean and unit variance for stable combination
    lcb_mean = np.mean(lcb_score)
    lcb_std = np.std(lcb_score)
    if lcb_std < 1e-12:
        lcb_norm = np.zeros_like(lcb_score)
    else:
        lcb_norm = (lcb_score - lcb_mean) / lcb_std
    
    # Combine normalized LCB with cost penalty
    scores = lcb_weight * lcb_norm + cost_weight * cost_penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
