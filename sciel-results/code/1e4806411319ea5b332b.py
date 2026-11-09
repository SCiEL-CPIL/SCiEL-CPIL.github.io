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
    
    # Structural Change: Non-linear scoring function
    # Standard Best Fit: score = -rem
    # This variant: score = -sqrt(rem + epsilon)
    # This increases the sensitivity to very small remaining capacities,
    # strongly favoring the tightest possible fit to minimize "dead" space.
    epsilon = 1e-6
    base = -np.sqrt(rem + epsilon)
    
    scores[feasible] = base
    
    if time.time() > deadline:
        return scores
        
    return scores
