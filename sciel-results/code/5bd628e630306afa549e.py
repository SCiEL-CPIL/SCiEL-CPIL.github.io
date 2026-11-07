"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: return simple cost-based scores if we're running out of time
    elapsed = time.time() - _start_time
    if elapsed > 280:  # Safety margin before 300s limit
        # Fallback: prefer low cost, high uncertainty
        return sigma / (cost + 1e-9)
    
    # Clip sigma to avoid division issues
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Use Cost-Adjusted Lower Confidence Bound (LCB)
    # LCB = mu - beta * sigma (we want to minimize, so lower is better)
    # But we need higher scores to be better, so we negate: -LCB = -mu + beta * sigma
    # Cost-adjusted: (-mu + beta * sigma) / cost
    
    # Beta parameter controls exploration-exploitation tradeoff
    # For multimodal functions like Styblinski-Tang, we want more exploration
    beta = 2.0
    
    # Compute cost-adjusted LCB score (higher is better)
    lcb_score = (-mu + beta * sigma) / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(lcb_score), lcb_score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
