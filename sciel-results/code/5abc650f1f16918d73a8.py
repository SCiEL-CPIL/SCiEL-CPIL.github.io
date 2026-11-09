"""Cost-aware acquisition for rastrigin 2d BO."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite scores (higher = evaluate next).
    
    Uses Cost-Aware Expected Improvement (cEI) with an adaptive UCB exploration term.
    Score = (EI + beta * sigma) / cost
    """
    # Deadline guard
    if time.time() - _start_time > 320:
        return np.zeros_like(mu)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Expected Improvement
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues in norm.cdf/pdf for very large/small values
    z_clipped = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive exploration parameter
    # When f_best is large (far from 0), we should explore more.
    # When f_best is small (close to 0), we should exploit.
    # f_best is in objective units (rastrigin is >= 0).
    # Typical rastrigin values are 0 to ~100+.
    # Let's scale beta based on f_best.
    # If f_best is 100, beta should be higher. If f_best is 0.1, beta should be lower.
    # beta = 0.5 * (1 + f_best / 10) ? No, that increases exploration as we get worse?
    # Usually, we want to explore when we don't know where the minimum is (high f_best).
    # So beta should be larger when f_best is larger.
    
    # However, if f_best is very large, the EI term is already large.
    # The UCB term helps escape local optima.
    
    # Let's use a simple adaptive beta:
    # beta = 0.1 * np.sqrt(np.log(1 + f_best))
    # Or just a constant beta that is tuned.
    # Let's try beta = 0.2 * (1.0 + f_best / 50.0)
    
    # Actually, a common choice is beta = 0.5 * sigma_scale?
    # Let's stick to a simple adaptive scheme:
    # beta = 0.1 if f_best > 10 else 0.05
    # Let's make it continuous:
    beta = 0.1 * (1.0 + np.log1p(f_best))
    
    # Exploration term
    exploration = beta * sigma_safe
    
    # Total numerator
    numerator = ei + exploration
    
    # Cost-aware score
    score = numerator / cost_safe
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
