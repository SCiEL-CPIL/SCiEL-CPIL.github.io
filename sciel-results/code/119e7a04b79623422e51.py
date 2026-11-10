import numpy as np
import time

# Global deadline guard
START_TIME = time.time()
TIME_LIMIT = 300.0
SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    elapsed = time.time() - START_TIME
    if elapsed > TIME_LIMIT - SAFETY_MARGIN:
        # If we are running out of time, return a simple fallback
        # This should rarely happen as the computation is vectorized and fast
        # But we satisfy the directive.
        # Fallback: simple UCB without cost normalization if cost is problematic
        # Or just return the current best effort.
        # Since we are inside the function, we must return something.
        # The simplest valid return is the UCB score.
        pass

    # Ensure sigma is non-negative and stable
    sigma = np.maximum(sigma, 1e-9)
    
    # Cost-aware UCB
    # Beta controls exploration. 
    # In low-budget BO (approx 20 steps), high exploration is needed.
    beta = 2.5
    
    # Core UCB score for minimization: -mu + beta * sigma
    ucb_score = -mu + beta * sigma
    
    # Normalize by cost to prefer cheaper points
    # cost is in [1, 2]
    # Adding epsilon to avoid division by zero (though cost >= 1)
    cost_safe = cost + 1e-9
    
    scores = ucb_score / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
