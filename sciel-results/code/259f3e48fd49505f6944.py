# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global timer for safety
_t0 = time.time()
_DEADLINE = 320.0

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() - _t0 > _DEADLINE:
        return -mu

    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Use LCB with adaptive beta
    # beta should be larger early (more exploration), smaller late (more exploitation)
    # f_best tells us how close we are to optimum (0)
    # When f_best is large, explore more; when f_best is small, exploit more
    
    # Adaptive beta: starts around 2.0, decreases as f_best approaches 0
    # Use a smooth transition
    f_best_clipped = np.clip(f_best, 1e-6, 10.0)
    beta = 2.0 / (1.0 + f_best_clipped)
    
    # LCB for minimization: mu - beta * sigma
    lcb = mu - beta * sigma_safe
    
    # Ensure finite scores
    lcb = np.where(np.isfinite(lcb), lcb, -1e10)
    
    return lcb
# EVOLVE-BLOCK-END
