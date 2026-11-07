"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Beale 2D.
    
    Uses an adaptive Lower Confidence Bound (LCB) where the exploration
    parameter beta scales with the inverse of the current best value.
    When f_best is large (far from minimum), beta is large -> more exploration.
    When f_best approaches 0, beta shrinks -> more exploitation.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s limit

    n = len(mu)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-9)
    mu = np.asarray(mu, dtype=np.float64)
    
    # Ensure f_best is finite
    if not np.isfinite(f_best):
        f_best = 1e6
    
    # Adaptive beta: scales with 1/f_best to prioritize exploration when far from minimum
    # beta = 1 + 10 / max(|f_best|, 1e-6)
    # When f_best is large (e.g., 100), beta ≈ 1.1 (moderate exploration)
    # When f_best is small (e.g., 0.1), beta ≈ 101 (high exploration)
    # Wait, that's backwards. Let me reconsider.
    
    # Actually, we want:
    # - When f_best is far from 0 (large), we should explore MORE to find the basin
    # - When f_best is close to 0, we should exploit MORE to refine the minimum
    
    # So beta should be LARGE when f_best is LARGE, and SMALL when f_best is SMALL.
    # beta = 1 + 10 * f_best / (1 + f_best) would give beta in [1, 11]
    # Or simpler: beta = 1 + 10 / (1 + 1/f_best) for f_best > 0
    # Let's use: beta = 1 + 10 * min(f_best / 10, 1.0) for f_best > 0
    # This gives beta in [1, 11], with beta=11 when f_best >= 10
    
    abs_f_best = abs(f_best)
    if abs_f_best < 1e-6:
        beta = 1.0
    else:
        # Scale beta from 1 to ~11 as f_best goes from 0 to 10+
        beta = 1.0 + 10.0 * min(abs_f_best / 10.0, 1.0)
    
    # Compute LCB: mu - beta * sigma (we want to MINIMIZE this, so we score by negating or using as-is)
    # For maximization-based selection, we want higher scores to mean "evaluate next"
    # LCB for minimization: lower LCB is better, so score = -LCB = -mu + beta*sigma
    # Or equivalently, score = beta * sigma - mu
    
    lcb_scores = beta * sigma - mu
    
    # Add a small EI component for stability
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Normalize EI to [0, 1]
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ei_norm = ei / ei_max
    
    # Combine: primarily LCB, with a small EI boost
    final_scores = lcb_scores + 0.1 * ei_max * ei_norm
    
    # Ensure finite values
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check time
    if time.time() > deadline:
        # Return the result as-is; it should be ready
        pass
    
    return final_scores
# EVOLVE-BLOCK-END
