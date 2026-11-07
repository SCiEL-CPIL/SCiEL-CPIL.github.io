import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 325.0

def acquisition(mu, sigma, f_best):
    # Check time
    if time.time() > _DEADLINE:
        # Return a valid array to avoid crash, though we might be killed anyway
        # Just return zeros or something simple
        return np.zeros_like(mu)
    
    # LCB acquisition function for minimization
    # We want to minimize mu - beta * sigma
    # So we maximize beta * sigma - mu
    
    # Beta controls exploration. 
    # 2.0 is a standard choice for 1D problems with limited budget.
    beta = 2.0
    
    scores = beta * sigma - mu
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
