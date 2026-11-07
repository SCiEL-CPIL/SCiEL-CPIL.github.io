"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Uses a hybrid approach:
    - Expected Improvement (EI) for exploitation
    - Normalized UCB-like exploration term with adaptive beta
    - As f_best improves, shift from exploration to exploitation
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety margin for sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (for minimization, we want to improve f_best)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize EI to [0, 1]
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    if ei_max > ei_min:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei)
    
    # Compute normalized exploration term (UCB-like)
    # We want to reward high sigma regions
    sigma_max = np.max(sigma_safe)
    sigma_min = np.min(sigma_safe)
    if sigma_max > sigma_min:
        exploration = (sigma_safe - sigma_min) / (sigma_max - sigma_min)
    else:
        exploration = np.zeros_like(sigma_safe)
    
    # Adaptive beta: start with strong exploration, shift to exploitation
    # f_best starts high (poor) and decreases toward global_min
    # progress: 0 when f_best is poor, 1 when f_best is near global_min
    global_min = -1.8013
    # Assume typical f_best range is [-2.0, 1.0]
    progress = np.clip((f_best - 1.0) / (global_min - 1.0), 0.0, 1.0)
    
    # Beta decreases from 2.0 (exploration) to 0.5 (exploitation)
    beta = 2.0 - 1.5 * progress
    
    # Weight for EI increases with progress
    w_ei = 0.3 + 0.7 * progress  # 0.3 early, 1.0 late
    w_explore = 1.0 - w_ei  # 0.7 early, 0.0 late
    
    # Blend: weighted sum of normalized EI and exploration
    score = w_ei * ei_norm + w_explore * beta * exploration
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
