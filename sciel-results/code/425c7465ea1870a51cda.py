# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Expected improvement per unit cost, with softened cost penalty.
    # Cost is in [1, 2]. Normalized cost impact is reduced.
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Soften the cost division. 
    # Original: 1/cost. Range [0.5, 1.0].
    # New: 1 / (1 + 0.5 * (cost - 1)). Range [1.0, 0.66].
    # This reduces the bias against higher-cost regions (which might contain the minimum).
    cost_penalty = 1.0 / (1.0 + 0.5 * (cost - 1.0))
    
    return ei * cost_penalty
# EVOLVE-BLOCK-END
