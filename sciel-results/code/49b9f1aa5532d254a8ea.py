import numpy as np
import time

# Global variable to track start time for the deadline guard
_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    
    # Initialize start time on first call
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline Guard: 
    # The evaluator kills after 330s. We have 30 evaluations.
    # We check if we are dangerously close to the limit.
    # If so, we return a "safe" acquisition function that just exploits 
    # the most promising point (lowest mu) to finish quickly and safely.
    # However, since the computation is very fast, this is likely never triggered.
    # But we must include it per directive.
    elapsed = time.time() - _start_time
    # Assume a safety margin of 10 seconds before the 330s limit
    if elapsed > 320: 
        # Return score based purely on exploitation (lowest mu is best)
        # Score = -mu (higher is better)
        return -mu
    
    # Standard UCB acquisition for minimization
    # We want to maximize the score.
    # UCB_min = mu - beta * sigma
    # To make "higher is better", we can return -UCB_min = beta * sigma - mu
    # Or simply return a score proportional to potential improvement.
    
    # Beta parameter for exploration-exploitation trade-off.
    # 1.0 is a standard default. 
    beta = 1.0
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate UCB score
    # score = beta * sigma - mu
    # This ensures that points with low mean and high uncertainty get high scores.
    scores = beta * sigma_safe - mu
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
