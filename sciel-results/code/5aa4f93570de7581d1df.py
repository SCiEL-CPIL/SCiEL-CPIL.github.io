import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()

def acquisition(mu, sigma, f_best, cost):
    elapsed = time.time() - _START_TIME
    if elapsed > 290:
        return -mu

    eps = 1e-9
    s = np.maximum(sigma, eps)
    c = np.maximum(cost, eps)

    total_cost = np.sum(cost)
    budget_progress = np.clip(total_cost / 30.0, 0.0, 1.0)

    # Dynamic cost penalty exponent
    # Start with lower penalty (alpha=0.5) to allow more exploration
    # End with higher penalty (alpha=1.0) to conserve budget
    alpha = 0.5 + 0.5 * budget_progress
    
    # Calculate cost factor
    # Using power law instead of linear division
    cost_factor = 1.0 / (c ** alpha)
    
    # Standard EI
    z = (f_best - mu) / s
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + s * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Exploration floor
    # Increased initial weight from 0.5 to 1.0
    exploration_weight = 1.0 * (1.0 - budget_progress)
    exploration_floor = exploration_weight * s * cost_factor
    
    # Combined Score
    score = ei * cost_factor + exploration_floor
    
    # Clean up
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
