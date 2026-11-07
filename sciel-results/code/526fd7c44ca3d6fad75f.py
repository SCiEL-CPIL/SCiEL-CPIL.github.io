"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    """
    Compute heuristic matrix for ACO on TSP.
    Uses a power of inverse distance to emphasize shorter edges more strongly.
    """
    # Add small epsilon to avoid division by zero
    dm = distance_matrix + 1e-9
    
    # Use a power of 2 to emphasize shorter distances more strongly
    # This helps guide ants towards shorter edges early in the process
    heuristic = 1.0 / (dm ** 2)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
