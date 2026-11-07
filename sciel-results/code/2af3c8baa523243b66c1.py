import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = _start_time + 280  # 20s safety margin before 300s kill

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() > _DEADLINE:
        # Fallback: simple cost-aware score
        return (f_best - mu) / (cost + 1e-9)

    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-9)
    
    # Cost-Effective UCB (Upper Confidence Bound)
    # For minimization, we want to minimize mu - beta * sigma
    # So the "improvement" potential is f_best - (mu - beta * sigma)
    # = f_best - mu + beta * sigma
    
    beta = 2.0  # Exploration-exploitation trade-off parameter
    
    # Calculate the UCB-based improvement potential
    # If f_best is very high initially, this will be large.
    # As f_best decreases, the numerator shrinks for points with high mu.
    ucb_improvement = f_best - mu + beta * sigma_safe
    
    # Cost-aware score
    # We want to maximize this score.
    # Higher ucb_improvement is better. Lower cost is better.
    scores = ucb_improvement / (cost + 1e-9)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # If all scores are -inf (should not happen), return zeros
    if np.all(scores == -np.inf):
        return np.zeros_like(mu)
        
    return scores
