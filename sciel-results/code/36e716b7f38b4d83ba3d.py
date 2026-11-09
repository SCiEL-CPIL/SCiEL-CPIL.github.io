"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Probability of Improvement (POI) with adaptive exploration, cost-normalized.
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # POI: probability that f(x) < f_best
    poi = norm.cdf(z)
    
    # Adaptive exploration: normalize sigma to [0,1] range
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-9:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Small exploration weight
    lambda_explore = 0.1
    combined = poi + lambda_explore * sigma_norm
    
    # Cost-aware: divide by cost (higher cost = lower priority)
    score = combined / np.maximum(cost, 1e-9)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
