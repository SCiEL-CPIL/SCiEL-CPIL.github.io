"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Deadline guard: return fallback if we're getting close to timeout
    deadline = start_time + 280  # 280s margin from 300s limit
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute Expected Improvement per unit cost
    ei_cost = ei / cost
    
    # Compute UCB-style score for exploration
    # UCB: f_best - mu + kappa * sigma
    kappa = 1.5
    ucb = (f_best - mu) + kappa * sigma_safe
    ucb_cost = ucb / cost
    
    # Blend EI and UCB: primarily EI with some UCB for exploration
    # The blend weight is tuned to balance exploitation and exploration
    score = 0.85 * ei_cost + 0.15 * ucb_cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 280:
        # Return EI cost as fallback
        ei_cost = np.where(np.isfinite(ei_cost), ei_cost, 0.0)
        return ei_cost
    
    return score
# EVOLVE-BLOCK-END
