"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # Calculate total prize for each item (sum across the 5 dimensions)
    total_prizes = np.sum(prizes, axis=0)
    
    # Calculate the maximum weight for each item across the 5 constraints
    # This identifies the "bottleneck" constraint for each item
    max_weights = np.max(weights, axis=0)
    
    # Heuristic: Prize divided by the maximum weight component
    # This favors items that are valuable and do not heavily strain any single constraint
    heur = total_prizes / (max_weights + 1e-9)
    
    # Ensure positive values as per contract
    heur[heur <= 0] = 1e-9
    
    return heur
# EVOLVE-BLOCK-END
