"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    # prizes and weights are (5, 100)
    # Calculate the maximum weight for each item across all constraints
    # This represents the "bottleneck" weight of the item
    max_weights = np.max(weights, axis=0)
    
    # Heuristic: Prize per unit of bottleneck weight
    # Items that are small in their most restrictive dimension are preferred
    # This encourages packing items that don't exhaust any single constraint prematurely
    epsilon = 1e-9
    eta = prizes / (max_weights + epsilon)
    
    # Ensure positivity as per contract (though prizes > 0 and weights > 0 usually)
    eta[eta <= 0] = 1e-9
    
    return eta
# EVOLVE-BLOCK-END
