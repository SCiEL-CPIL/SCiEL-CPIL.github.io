import numpy as np
import time

# Global state for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: 300s limit, leave 30s margin
    if time.time() - _start_time > 270:
        # Fallback: Return simple cost-adjusted negative mean (exploitation only)
        # Ensure finite
        return np.where(np.isfinite(mu), -mu / (cost + 1e-9), 0.0)

    # Calculate progress based on f_best
    # 3HC min is 0. Values can be up to ~10.
    # Clip f_best to [0, 10] to estimate progress
    f_best_val = float(np.min(f_best))
    
    progress = np.clip(f_best_val, 0.0, 10.0) / 10.0
    
    # Beta decays from 2.5 (early) to 0.3 (late) for stronger exploration early
    beta = 0.3 + 2.2 * progress
    
    # Calculate Lower Confidence Bound (for minimization)
    # LCB = mu - beta * sigma
    # We want to minimize LCB / cost.
    # So we maximize -LCB / cost.
    
    # Handle sigma=0
    sigma_safe = np.maximum(sigma, 1e-9)
    
    lcb = mu - beta * sigma_safe
    
    # Cost penalty: use 1/cost for stronger differentiation
    # cost in [1, 2], so 1/cost in [0.5, 1.0]
    cost_penalty = 1.0 / cost
    
    score = -lcb * cost_penalty
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
