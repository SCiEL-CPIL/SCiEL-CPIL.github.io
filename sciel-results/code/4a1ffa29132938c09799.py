import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    t0 = time.time()
    deadline = t0 + 115.0
    
    remaining = bins - item
    feasible = remaining >= 0
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(feasible):
        return scores
    
    rem = remaining[feasible]
    
    # Refined non-linear scoring: blend of power and linear terms
    # Use a softer power (0.75) to reduce over-penalization of small remainders
    # Add a small linear component to maintain sensitivity across the range
    epsilon = 1e-6
    power_score = -np.power(rem + epsilon, 0.75)
    linear_score = -0.1 * rem
    base = power_score + linear_score
    
    scores[feasible] = base
    
    if time.time() > deadline:
        return scores
        
    return scores
