"""Cost-aware acquisition for BO on four benchmark functions."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 295.0  # Safety margin before 300s/330s limit

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT:
        return np.zeros_like(mu)
    
    # Ensure sigma is positive for division
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    delta = f_best - mu
    z = delta / sigma_safe
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = delta * cdf + sigma_safe * pdf
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Cost-aware score: EI per unit cost
    # Add a small exploration bonus proportional to sigma to encourage
    # visiting uncertain regions, scaled by 1/cost to maintain cost-awareness
    exploration_weight = 0.1
    exploration_bonus = exploration_weight * sigma_safe / cost_safe
    
    score = ei / cost_safe + exploration_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
