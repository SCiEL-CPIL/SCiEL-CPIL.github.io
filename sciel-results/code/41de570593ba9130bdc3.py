"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: 330s wall-clock limit, return safely before it.
    t0 = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero or near-zero sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Linear regret lower bound: f_best - mu + sigma
    # This is a simpler, smoother acquisition than EI
    # It balances exploitation (f_best - mu) and exploration (sigma)
    linear_regret = (f_best - mu) + sigma_safe
    
    # Cost-aware: divide by cost to get improvement per unit cost
    scores = linear_regret / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time: if we've used more than 325s, return immediately
    if time.time() - t0 > 325:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
