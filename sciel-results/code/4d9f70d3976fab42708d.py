"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    total_prize = prizes.sum(axis=0)  # (100,)
    
    # Tightness of each constraint: total demand / capacity
    total_demand = weights.sum(axis=1)  # (5,)
    tightness = total_demand / 25.0  # (5,)
    
    # Weighted cost: items consuming tight constraints are penalized more
    weighted_cost = (tightness[:, None] * weights).sum(axis=0)  # (100,)
    
    # Heuristic: total prize / weighted cost
    heuristic = total_prize / (weighted_cost + 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
