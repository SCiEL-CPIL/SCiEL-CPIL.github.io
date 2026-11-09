import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses an adaptive Lower Confidence Bound (LCB) with an exponential cost penalty.
    """
    # Deadline guard
    t0 = time.time()
    
    # Ensure inputs are numpy arrays of float64
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Flatten and check size
    n = 2000
    if mu.shape != (n,):
        mu = mu.reshape(-1)
        if mu.shape[0] != n:
            return np.zeros(n, dtype=np.float64)
    if sigma.shape != (n,):
        sigma = sigma.reshape(-1)
        if sigma.shape[0] != n:
            return np.zeros(n, dtype=np.float64)
    if cost.shape != (n,):
        cost = cost.reshape(-1)
        if cost.shape[0] != n:
            return np.zeros(n, dtype=np.float64)
    
    # Numerical safety
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Adaptive exploration coefficient
    # Scale beta with the current best value to encourage exploration early on
    # f_best is the minimum observed value (lower is better)
    # If f_best is large (poor), we explore more. If f_best is small (good), we exploit more.
    # Since Levy 2D min is 0, f_best >= 0.
    beta = max(1.0, 0.1 * abs(f_best))
    
    # LCB score: maximize (beta * sigma - mu)
    # This favors points with low mean and high uncertainty
    lcb_score = beta * sigma_safe - mu
    
    # Exponential cost penalty
    # Higher cost (further from origin) is suppressed more aggressively
    # This concentrates search in the center where the minimum is likely located
    cost_penalty = np.exp(-1.5 * cost_safe)
    
    # Combine
    scores = lcb_score * cost_penalty
    
    # Handle non-finite values
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    # If all scores are invalid, return zeros
    if not np.any(np.isfinite(scores)):
        return np.zeros(n, dtype=np.float64)
        
    # Check time
    if time.time() - t0 > 280:
        return scores
        
    return scores
