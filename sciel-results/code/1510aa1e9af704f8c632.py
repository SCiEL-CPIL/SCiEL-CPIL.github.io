import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 280:
        # Fallback to simple EI/cost
        z = (f_best - mu) / (sigma + 1e-9)
        z = np.clip(z, -30, 30)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / cost
    
    # UCB with cost awareness
    # Beta controls exploration vs exploitation
    # Higher beta -> more exploration
    beta = 2.0
    
    # Calculate UCB: mu + beta * sigma
    ucb = mu + beta * sigma
    
    # Cost-aware score: UCB / cost
    # We want to maximize this
    scores = ucb / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
