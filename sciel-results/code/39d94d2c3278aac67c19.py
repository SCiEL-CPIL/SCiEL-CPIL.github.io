"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: 330s wall-clock limit, use 300s for safety
    t_start = time.time()
    deadline = t_start + 295.0
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    n = len(mu)
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Avoid division by zero in cost
    cost_safe = np.where(cost > 1e-9, cost, 1.0)
    
    # Adaptive LCB with dynamic beta
    # beta scales with relative uncertainty to balance exploration/exploitation
    # Use a base beta that adapts based on the spread of sigma relative to f_best
    sigma_mean = np.mean(sigma)
    sigma_max = np.max(sigma)
    f_best_val = f_best
    
    # Dynamic beta: start moderate, increase if uncertainty is high
    # This helps avoid premature convergence (EI's weakness)
    # beta in range [0.5, 2.0] typically works well
    if sigma_max < 1e-6:
        beta = 0.5
    else:
        # Scale beta with relative uncertainty
        rel_uncert = sigma_mean / (abs(f_best_val) + 1e-9)
        beta = 0.5 + 1.5 * min(rel_uncert / (abs(f_best_val) + 1e-9), 1.0)
        beta = np.clip(beta, 0.5, 2.0)
    
    # LCB: mu - beta * sigma, normalized by cost
    # Lower LCB = more promising (we want to minimize)
    # So score = -LCB / cost = (beta * sigma - mu) / cost
    # Higher score = better
    
    # Check deadline
    if time.time() > deadline:
        # Return simple EI-based scores as fallback
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        scores = ei / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Compute LCB-based score
    lcb = mu - beta * sigma
    scores = -lcb / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline again
    if time.time() > deadline:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
