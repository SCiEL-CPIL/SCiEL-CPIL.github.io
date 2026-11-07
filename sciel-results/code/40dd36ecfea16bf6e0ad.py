# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_DEADLINE = 325.0  # safety margin before 330s kill
_T0 = None


def acquisition(mu, sigma, f_best, cost):
    global _T0
    if _T0 is None:
        _T0 = time.time()
    
    # Deadline guard: if we're running long, return a simple score quickly
    if time.time() - _T0 > _DEADLINE:
        return np.where(sigma > 1e-8, (f_best - mu) / (sigma + 1e-9), 0.0) / (cost + 1e-9)
    
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Standard EI with exploration offset xi
    xi = 0.05  # exploration offset, encourages sampling uncertain regions
    improvement = f_best - mu - xi
    z = improvement / sigma_safe
    
    # EI = (f_best - mu - xi) * Phi(z) + sigma * phi(z)
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a pure exploration term to help escape local minima of rastrigin
    # This is important because rastrigin has many local minima
    exploration_weight = 0.5
    exploration = exploration_weight * sigma_safe
    
    # Combined acquisition: (EI + exploration) / cost
    score = (ei + exploration) / (cost + 1e-9)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
