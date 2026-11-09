"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # Calculate total prize for each item (sum across the 5 dimensions)
    total_prizes = np.sum(prizes, axis=0)
    
    # Calculate the maximum weight for each item across the 5 constraints
    max_weights = np.max(weights, axis=0)
    
    # Calculate the sum of weights for each item across the 5 constraints
    # This captures the total "bulk" of the item
    sum_weights = np.sum(weights, axis=0)
    
    # Heuristic: Prize divided by a composite weight penalty
    # We penalize the maximum weight (bottleneck) and the total weight (bulk)
    # The coefficient 0.5 balances the two effects
    composite_weight = max_weights + 0.5 * sum_weights
    
    heur = total_prizes / (composite_weight + 1e-9)
    
    # Ensure positive values as per contract
    heur[heur <= 0] = 1e-9
    
    return heur
# EVOLVE-BLOCK-END
