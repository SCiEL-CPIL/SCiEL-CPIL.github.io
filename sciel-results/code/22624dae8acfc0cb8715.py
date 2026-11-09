import numpy as np
import time

# Global deadline management
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0 # 10s margin

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a safe default, e.g., uniform scores or based on mu only
        # To avoid invalidating, return finite scores.
        # A simple fallback: -mu / cost
        cost_safe = np.maximum(cost, 1e-9)
        fallback_scores = -mu / cost_safe
        return np.where(np.isfinite(fallback_scores), fallback_scores, 0.0)

    # Safe sigma and cost
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Lower Confidence Bound (LCB) for minimization
    # We want to minimize f(x).
    # LCB(x) = mu(x) - beta * sigma(x)
    # Lower LCB is better.
    # We want to MAXIMIZE the score.
    # So Score = -LCB(x) / cost(x)
    
    beta = 1.5
    lcb = mu - beta * sigma_safe
    
    scores = -lcb / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
