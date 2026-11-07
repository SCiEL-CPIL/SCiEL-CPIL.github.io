"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # prizes: (5, 100), weights: (5, 100)
    # Total prize per item
    total_prize = np.sum(prizes, axis=0)
    
    # Total weight per item across all constraints
    total_weight = np.sum(weights, axis=0)
    
    # Base heuristic: prize-to-weight ratio with smoothing
    # The +1.0 smoothing (from inspiration 2) stabilizes the ratio
    # and prevents extreme values for very low-weight items
    eta = total_prize / (total_weight + 1.0)
    
    # Ensure positivity
    eta = np.maximum(eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
