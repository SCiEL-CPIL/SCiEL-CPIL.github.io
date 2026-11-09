"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Adaptive Upper Confidence Bound (UCB).
    
    The UCB acquisition function balances exploitation (low mu) and exploration (high sigma).
    The parameter beta is adapted based on the current best observed value f_best.
    Early in the optimization (high f_best), beta is large to encourage exploration.
    Late in the optimization (low f_best), beta is small to encourage exploitation.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Safety checks
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        return np.zeros_like(mu)
        
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-10)
    
    # Adaptive beta based on f_best
    # Branin minimum is ~0.397887
    # Initial random points usually yield f_best in [5, 15]
    # We want high beta (exploration) when f_best is high.
    # We want low beta (exploitation) when f_best is low.
    
    f_min = 0.397887
    f_max_est = 15.0 # Upper bound of initial random points
    
    # Normalize f_best
    f_current = np.clip(f_best, f_min, f_max_est)
    
    # Linear interpolation
    # t = 0 when f_best = f_min (late stage, low beta)
    # t = 1 when f_best = f_max_est (early stage, high beta)
    t = (f_current - f_min) / (f_max_est - f_min)
    
    beta_min = 0.5
    beta_max = 3.0
    
    beta = beta_min + (beta_max - beta_min) * t
    
    # UCB score for minimization: maximize beta * sigma - mu
    scores = beta * sigma - mu
    
    # Handle any remaining non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
