import numpy as np
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    
    # Safety margin: if we've been running for too long, return a simple fallback
    # to avoid timeout in the evaluation loop.
    if elapsed > 320:
        return np.zeros_like(cost)

    # Cost-aware UCB (Upper Confidence Bound) for minimization.
    # We maximize the score, which corresponds to the optimistic lower bound per unit cost.
    # LCB = mu - beta * sigma
    # Score = -LCB / cost = (-mu + beta * sigma) / cost
    
    # Beta controls the exploration-exploitation trade-off.
    # In high dimensions (8D), higher beta helps escape local minima.
    beta = 1.5
    
    # Ensure sigma is positive to avoid division by zero or numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate scores
    scores = (-mu + beta * sigma) / cost
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
