"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # prizes and weights are of shape (5, 100)
    # We need to return an array of shape (100,)
    
    # 1. Calculate total prize per item (sum across the 5 constraint rows)
    total_prizes = np.sum(prizes, axis=0)
    
    # 2. Identify the most constraining weight per item.
    # An item is limited by the constraint where its weight is highest.
    # Using max instead of sum correctly penalizes items that are heavy in any single constraint,
    # which is more critical for feasibility than the average weight.
    max_weights = np.max(weights, axis=0)
    
    # 3. Compute raw heuristic: Value Density based on the bottleneck constraint
    # Items with high prize and low bottleneck weight are preferred.
    raw_heuristic = total_prizes / (max_weights + 1e-9)
    
    # 4. Normalize the heuristic values.
    # Scaling so the best item has a heuristic of 1.0 helps balance the
    # influence of pheromone and heuristic in the ACO probability calculation.
    max_heuristic = np.max(raw_heuristic)
    if max_heuristic > 1e-9:
        normalized_heuristic = raw_heuristic / max_heuristic
    else:
        # Fallback if all items are zero or invalid (should not happen with uniform [0,1])
        normalized_heuristic = np.ones_like(raw_heuristic)
        
    return normalized_heuristic
# EVOLVE-BLOCK-END
