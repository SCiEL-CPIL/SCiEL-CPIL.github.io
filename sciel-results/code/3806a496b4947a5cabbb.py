"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    # Calculate sum of pairs
    pair = demands[:, None] + demands[None, :]
    
    # Mask for pairs that fit in the bin
    mask = pair <= capacity
    
    # Calculate ratio of fill for fitting pairs
    # Avoid division by zero or invalid operations on non-fitting pairs
    ratio = np.zeros_like(pair)
    ratio[mask] = pair[mask] / capacity
    
    # Apply a strong power to favor tight fits
    # This encourages the ACO to behave more like a greedy Best-Fit algorithm
    # which is known to perform well on bin packing.
    power = 10.0
    heuristic = np.where(mask, ratio ** power, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
