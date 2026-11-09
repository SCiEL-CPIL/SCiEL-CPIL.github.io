import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _start_time > 320:
        return np.zeros_like(mu)

    sigma = np.maximum(sigma, 1e-6)
    
    # Cost-Aware Lower Confidence Bound (LCB)
    # For minimization, we maximize -mu + beta * sigma
    # Scale exploration by 1/sqrt(cost) to penalize expensive evaluations
    cost_safe = np.maximum(cost, 1e-6)
    cost_sqrt = np.sqrt(cost_safe)
    
    beta = 2.5
    exploration = beta * sigma / cost_sqrt
    exploitation = -mu
    
    scores = exploitation + exploration
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
