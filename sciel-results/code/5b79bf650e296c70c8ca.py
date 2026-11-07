import numpy as np
import time

# Module-level deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're running out of time, return a safe default
    if time.time() > _DEADLINE - 5.0:
        # Return a simple LCB score that's still valid
        sigma_safe = np.maximum(sigma, 1e-9)
        beta = 2.0
        score = mu - beta * sigma_safe
        return np.where(np.isfinite(score), score, 0.0)

    # Ensure sigma is non-negative and not zero
    sigma = np.maximum(sigma, 1e-9)
    
    # GP-LCB strategy:
    # Instead of cost-aware EI (which penalizes expensive points heavily),
    # we use a standard Lower Confidence Bound.
    # This treats all points equally in terms of cost, relying solely on the GP's
    # uncertainty (sigma) and mean (mu) to guide exploration.
    # This is a materially different design family: 
    # 1. No division by cost (avoids biasing towards low-cost corners).
    # 2. Explicit exploration parameter beta.
    # 3. Linear combination vs non-linear probabilistic expectation.
    
    beta = 2.0
    score = mu - beta * sigma
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
