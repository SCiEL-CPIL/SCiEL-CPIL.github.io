import numpy as np
import time

# Global state for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: 330s limit, return safe result if close to limit
    # Since we return an array, "best result found" implies we should return
    # a valid acquisition array. If we are running out of time, we can just
    # return a simple heuristic or the current calculation which is fast.
    # The computation is O(N) vectorized, so it's very fast.
    # We only need to worry if the loop itself is slow, but we don't control the loop.
    # We just return the scores.
    
    # Cost-aware UCB for minimization
    # Score = (f_best - mu + kappa * sigma) / cost
    
    # Use a kappa that balances exploration and exploitation.
    # 2.5 is a standard value for BO.
    kappa = 2.5
    
    # Ensure sigma is not zero to avoid numerical issues (though UCB doesn't divide by sigma)
    # But if sigma is 0, the term is just f_best - mu.
    
    # Calculate UCB
    # f_best is a scalar, mu and sigma are arrays
    ucb = f_best - mu + kappa * sigma
    
    # Normalize by cost
    # cost is an array
    # Avoid division by zero
    cost_safe = np.where(cost > 1e-8, cost, 1.0)
    
    scores = ucb / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
