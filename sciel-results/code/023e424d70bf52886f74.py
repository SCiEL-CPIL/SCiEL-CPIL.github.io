import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # Safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    # Check deadline - if too close, return a simple acquisition to avoid timeout
    if time.time() - _start_time > _DEADLINE:
        # Fallback: simple LCB/cost
        sigma_safe = np.maximum(sigma, 1e-9)
        kappa = 2.0
        lcb = mu - kappa * sigma_safe
        scores = -lcb / (cost + 1e-9)
        return np.where(np.isfinite(scores), scores, 0.0)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Cost-normalized Lower Confidence Bound (LCB)
    # LCB = mu - kappa * sigma (lower bound on true value)
    # We want to minimize, so higher acquisition = more promising
    # Use -LCB / cost = (kappa * sigma - mu) / cost
    # This directly targets the most promising lower bound per unit cost
    kappa = 2.0
    lcb = mu - kappa * sigma_safe
    
    # Acquisition: negative LCB divided by cost
    # Higher values indicate more promising points per unit cost
    cost_safe = cost + 1e-9
    scores = -lcb / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
