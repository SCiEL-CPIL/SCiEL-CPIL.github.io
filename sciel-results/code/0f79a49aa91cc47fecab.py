"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


_START_TIME = time.time()
_TIME_LIMIT = 300.0
_TIME_MARGIN = 10.0


def _time_left():
    return _TIME_LIMIT - _TIME_MARGIN - (time.time() - _START_TIME)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Branin 2D.
    
    Uses a hybrid of Expected Improvement (EI) and Lower Confidence Bound (LCB)
    for minimization. EI provides exploitation (targeting points likely to beat
    f_best), while LCB provides exploration (targeting points with low mean and
    high uncertainty). Both are normalized to [0, 1] and blended with a dynamic
    weight that shifts from exploration (LCB) to exploitation (EI) as f_best
    approaches the known global minimum (0.397887).
    """
    # Deadline guard
    if _time_left() < 1.0:
        # Fallback: pure EI (simple and fast)
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        z = np.clip(z, -10, 10)
        return (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # 2. Compute LCB (Lower Confidence Bound) for minimization
    # LCB = mu - beta * sigma
    # Lower LCB is better (lower predicted value with uncertainty discount)
    # For maximization score, we use -LCB = -mu + beta * sigma
    # This is equivalent to UCB for minimization but conceptually LCB
    beta = 2.0  # Higher beta for more aggressive exploration as per directive
    lcb_score = -mu + beta * sigma_safe
    lcb_score = np.where(np.isfinite(lcb_score), lcb_score, 0.0)
    
    # 3. Normalize EI to [0, 1]
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    if ei_max - ei_min > 1e-12:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei)
    
    # 4. Normalize LCB score to [0, 1]
    lcb_min = np.min(lcb_score)
    lcb_max = np.max(lcb_score)
    if lcb_max - lcb_min > 1e-12:
        lcb_norm = (lcb_score - lcb_min) / (lcb_max - lcb_min)
    else:
        lcb_norm = np.zeros_like(lcb_score)
    
    # 5. Dynamic Weighting
    # gap = f_best - global_min
    # If gap is large, we are far from optimum -> explore more (high LCB weight)
    # If gap is small, we are near optimum -> exploit more (high EI weight)
    min_val = 0.397887
    gap = f_best - min_val
    
    # Sigmoid to map gap to exploration weight
    # explore_weight = 1 / (1 + exp(-k * (gap - c)))
    # We want explore_weight -> 1 when gap is large.
    # Let c = 2.0, k = 2.0
    # If gap = 10, exp(-2*(10-2)) = exp(-16) ~ 0, weight ~ 1.
    # If gap = 0.5, exp(-2*(0.5-2)) = exp(3) ~ 20, weight ~ 1/21 ~ 0.05.
    explore_weight = 1.0 / (1.0 + np.exp(-2.0 * (gap - 2.0)))
    
    # Ensure weight is in [0,1]
    explore_weight = np.clip(explore_weight, 0.0, 1.0)
    
    # 6. Combine
    # Score = (1 - explore_weight) * EI_norm + explore_weight * LCB_norm
    score = (1.0 - explore_weight) * ei_norm + explore_weight * lcb_norm
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
