import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIMEOUT = 320.0

def acquisition(mu, sigma, f_best, cost):
    """
    Stochastic shortest-path inspired acquisition:
    Geometric mean of normalized EI and inverse cost.
    """
    if time.time() - _START_TIME > _TIMEOUT:
        return -cost

    n = len(mu)
    
    # 1. Calculate Expected Improvement (EI)
    sigma_safe = np.where(sigma < 1e-9, 1e-9, sigma)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 2. Normalize EI and Cost
    # EI: [0, max_ei] -> [0, 1]
    ei_max = np.max(ei)
    if ei_max < 1e-9:
        ei_norm = np.ones(n)
    else:
        ei_norm = ei / ei_max
        
    # Cost: [min_cost, max_cost] -> [0, 1]
    # We want higher score for lower cost.
    # So we use inverse cost normalized.
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    
    # To avoid division by zero and handle scaling:
    # Map cost to [0, 1] where 0 is best (min cost) and 1 is worst (max cost)
    if cost_max - cost_min < 1e-9:
        cost_norm = np.zeros(n)
    else:
        cost_norm = (cost - cost_min) / (cost_max - cost_min)
        
    # Inverse cost metric: 1 - cost_norm (higher is better)
    inv_cost_norm = 1.0 - cost_norm
    
    # 3. Geometric Mean
    # To avoid log(0), add epsilon
    eps = 1e-9
    score = np.sqrt((ei_norm + eps) * (inv_cost_norm + eps))
    
    # Optional: We could still apply a soft penalty for very low EI
    # But the geometric mean naturally handles the trade-off.
    # If EI is 0, score is ~0 (good, no improvement expected).
    # If Cost is high, inv_cost is low, score is low.
    
    return score
