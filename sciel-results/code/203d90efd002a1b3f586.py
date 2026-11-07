"""Cost-aware acquisition: simple EI/cost with numerical safety."""
import numpy as np
from scipy.stats import norm
import time

# EVOLVE-BLOCK-START

def _safe_div(a, b, eps=1e-12):
    return a / (b + eps)

def _ei(mu, sigma, f_best, eps=1e-9):
    """Standard Expected Improvement for minimization, with z-clipping for numerical safety."""
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    # Clip z to prevent numerical issues in cdf/pdf
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)

def acquisition(mu, sigma, f_best, cost):
    """Cost-aware Expected Improvement: EI / cost."""
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-6)
    
    ei = _ei(mu, sigma, f_best)
    score = ei / cost_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

# EVOLVE-BLOCK-END
