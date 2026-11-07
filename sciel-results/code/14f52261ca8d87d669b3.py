import numpy as np
import time

def heuristics(prizes, weights):
    start_time = time.time()
    
    # prizes: (5, 100)
    # weights: (5, 100)
    
    # Calculate total prize per item
    total_prize = np.sum(prizes, axis=0)
    
    # Calculate the maximum weight per item across all constraints.
    # This represents the "bottleneck" resource consumption for each item.
    # An item that is very heavy in one constraint is more "expensive" in terms of
    # feasibility than one that is spread out, even if total weight is similar.
    bottleneck_weight = np.max(weights, axis=0)
    
    # Heuristic: Total Prize / Bottleneck Weight
    # This structural change focuses on the most constrained dimension for each item.
    eta = total_prize / (bottleneck_weight + 1e-9)
    
    # Ensure positivity and finite values
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    # Deadline guard
    if time.time() - start_time > 290:
        return eta
        
    return eta
