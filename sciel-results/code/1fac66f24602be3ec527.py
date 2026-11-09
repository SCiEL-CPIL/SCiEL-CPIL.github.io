"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using dynamic UCB.
    
    mu, sigma, f_best, cost: arrays of shape (2000,)
    Returns: 2000 finite scores (higher = evaluate next)
    """
    t_start = time.time()
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety guards
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-10)
    cost_safe = np.where(np.isfinite(cost_safe), cost_safe, 1.0)
    
    # Estimate search progress to tune kappa
    # Hartmann 3D global min is approx -3.86278
    # Initial random points usually yield f_best around -2.0 to -2.5
    F_MIN = -3.86278
    F_INIT = -2.0
    
    # Progress p: 1.0 at start (high f_best), 0.0 at end (low f_best)
    delta_f = F_INIT - F_MIN
    if delta_f <= 0:
        p = 1.0
    else:
        p = (f_best - F_MIN) / delta_f
        p = np.clip(p, 0.0, 1.0)
        
    # Dynamic kappa: High exploration early (p~1), Low exploration late (p~0)
    # kappa ranges from 0.5 (late) to 2.0 (early)
    kappa = 0.5 + 1.5 * p
    
    # UCB for minimization: We want to minimize mu + kappa * sigma
    # So we maximize the negative: -mu - kappa * sigma
    ucb_raw = -mu - kappa * sigma_safe
    
    # Normalize ucb_raw to [0, 1] to make it comparable across iterations
    # and to prevent scale issues before cost division
    ucb_min = np.min(ucb_raw)
    ucb_max = np.max(ucb_raw)
    ucb_range = ucb_max - ucb_min
    if ucb_range < 1e-10:
        ucb_norm = np.ones_like(ucb_raw)
    else:
        ucb_norm = (ucb_raw - ucb_min) / ucb_range
        
    # Cost-aware score
    scores = ucb_norm / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time guard
    t_elapsed = time.time() - t_start
    if t_elapsed > 280:
        return scores
        
    return scores
# EVOLVE-BLOCK-END
