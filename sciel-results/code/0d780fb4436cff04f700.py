import numpy as np
import time
import math

_START_TIME = time.time()
_TIME_LIMIT = 295.0  # Safety margin for 300s limit

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback: Greedy exploitation
        return -mu / np.maximum(cost, 1e-9)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Cost-aware acquisition function
    # We want to balance exploration (uncertainty) and exploitation (low mean)
    # while accounting for cost (cheaper points are preferred)
    
    # The cost is 1 + mean(x scaled to [0, 1])
    # Since we don't have x, we use the cost array directly
    
    # Strategy: Adaptive LCB (Lower Confidence Bound) with cost normalization
    # LCB = mu - beta * sigma
    # We want to minimize LCB / cost (since lower is better for objective, higher is better for score)
    # So score = -LCB / cost = -(mu - beta * sigma) / cost
    
    # Adaptive beta based on f_best:
    # When f_best is high (early), explore more (higher beta)
    # When f_best is low (late), exploit more (lower beta)
    
    # f_best ranges from ~130 (worst) to 0 (best) for rosenbrock in [-2,2]^2
    # Use a smooth transition
    f_best_clamped = np.clip(f_best, 0.0, 200.0)
    
    # Beta starts high when f_best is high, decreases as f_best approaches 0
    # beta = base + scale / (1 + f_best)
    # At f_best=100: beta = 0.5 + 3.0/101 ≈ 0.53
    # At f_best=10: beta = 0.5 + 3.0/11 ≈ 0.77
    # At f_best=1: beta = 0.5 + 3.0/2 = 2.0
    # At f_best=0: beta = 0.5 + 3.0/1 = 3.5
    
    # Actually, let's think about this differently.
    # We want more exploration early (high f_best) and less later.
    # So beta should be HIGH when f_best is HIGH.
    
    # beta = 0.5 + 3.0 * (f_best / (f_best + 1.0))
    # At f_best=100: beta ≈ 0.5 + 3.0*0.99 = 3.47
    # At f_best=10: beta = 0.5 + 3.0*0.91 = 3.23
    # At f_best=1: beta = 0.5 + 3.0*0.5 = 2.0
    # At f_best=0: beta = 0.5
    
    beta = 0.5 + 3.0 * (f_best_clamped / (f_best_clamped + 1.0))
    
    # LCB = mu - beta * sigma
    lcb = mu - beta * sigma
    
    # Score = -lcb / cost (higher is better)
    scores = -lcb / cost_safe
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
