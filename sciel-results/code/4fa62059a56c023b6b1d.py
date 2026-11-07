"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    """
    Returns an n x n matrix of attractiveness for placing item j right after item i
    in the same bin. Larger values are more attractive.
    """
    demands = np.asarray(demands, dtype=np.float64)
    n = len(demands)
    
    # Compute pair sums
    pair = demands[:, None] + demands[None, :]
    
    # Compute remaining capacity after placing i
    rem_i = capacity - demands
    rem_j = capacity - demands
    
    # Compute the fill ratio when j is placed after i in the same bin
    # This is (demands[i] + demands[j]) / capacity
    fill = pair / capacity
    
    # For pairs that fit (pair <= capacity), we want high attractiveness
    # for tight fits (high fill ratio)
    # Use fill^2 + 0.1 to reward tight fits while maintaining a positive baseline
    
    # Create mask for fitting pairs
    fits = pair <= capacity
    
    # For fitting pairs, use fill^2 + 0.1
    # For non-fitting pairs, use a very small value
    heuristic = np.where(fits, fill**2 + 0.1, 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
