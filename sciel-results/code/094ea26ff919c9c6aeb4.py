"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    # Use a stronger emphasis on high-prize nodes and a softer distance penalty
    # This helps the ACO find high-value routes within the length constraint
    prize_factor = np.power(prizes + 1e-9, 2.0)
    dist_factor = np.power(distance_matrix + 1e-3, 0.5)
    
    heuristic = prize_factor[None, :] / dist_factor
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
