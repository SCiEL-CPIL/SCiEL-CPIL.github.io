import numpy as np
import time

# Global start time for deadline guard
_start_time = time.time()
TIME_LIMIT = 300.0 # 300s limit per evaluation, but we are just one step.
# The evaluator runs the loop. The acquisition is called 25 times (30 evals - 5 init).
# 300s is per "evaluation" step? "Each evaluation must finish within 300 s."
# This likely means the whole step (GP fit + acquisition + selection) must be < 300s.
# Our function is just the acquisition. It should be milliseconds.

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Adaptive Upper Confidence Bound (UCB) for minimization.
    """
    # Deadline guard: if we are running out of time, return a safe fallback
    # However, since this function is very fast, this is mostly a precaution.
    if time.time() - _start_time > TIME_LIMIT - 5.0:
         # Return scores that prioritize exploitation of current best mean
         # to ensure we don't waste time on complex calculations (though none are left)
         sigma_safe = np.maximum(sigma, 1e-9)
         return -mu + 0.1 * sigma_safe

    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-9)
    
    # Adaptive exploration parameter beta
    # If f_best is large (poor), explore more (higher beta).
    # If f_best is small (good), exploit more (lower beta).
    # Weierstrass min is 0.
    beta = 1.0 + 4.0 * max(f_best, 0.0)
    
    # Calculate UCB scores
    # Higher score means evaluate next.
    scores = -mu + beta * sigma_safe
    
    # Ensure output is finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
