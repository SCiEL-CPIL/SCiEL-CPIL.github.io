"""Cost-aware acquisition for BO on three hump camel 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_TIME_LIMIT = 320.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement with
    adaptive exploration and softer cost penalty.
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        return np.zeros(len(mu), dtype=np.float64)
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1.0)
    
    # Expected Improvement for minimization
    Z = (f_best - mu) / sigma_safe
    Z = np.clip(Z, -10.0, 10.0)
    
    phi = norm.pdf(Z)
    Phi = norm.cdf(Z)
    
    EI = (f_best - mu) * Phi + sigma_safe * phi
    EI = np.maximum(EI, 0.0)
    
    # Adaptive Exploration
    # Scale exploration with the typical uncertainty
    median_sigma = np.median(sigma_safe)
    
    # Exploration weight: 
    # Higher when uncertainty is high.
    # We use a fixed multiplier on the median sigma to ensure exploration 
    # is comparable to the EI scale in early stages.
    alpha = 0.5
    
    exploration = alpha * median_sigma * sigma_safe
    
    # Cost penalty: 
    # Using a power of 0.5 (square root) reduces the penalty for expensive 
    # regions, allowing the optimizer to find minima that might be slightly 
    # more expensive but have much lower function values.
    cost_penalty = cost_safe ** 0.5
    
    # Base score
    score = (EI + exploration) / cost_penalty
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
