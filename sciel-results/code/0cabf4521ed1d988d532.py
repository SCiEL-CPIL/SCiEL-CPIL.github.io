"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Computes the heuristic matrix for Ant Colony Optimization.
    
    Uses an exponential decay model based on the total path extension cost.
    This approach avoids the singularity issues of rational heuristics and
    provides a smooth, bounded preference for nodes that are close to the
    current node and close to the start, weighted by their prize value.
    
    Formula: eta[i, j] = prize[j] * exp(-alpha * (distance[i, j] + distance[j, 0]))
    """
    prizes = np.asarray(prizes)
    distance_matrix = np.asarray(distance_matrix)
    
    # Ensure shapes are compatible
    n = prizes.shape[0]
    
    # distance[j, 0] is the cost to return to start from j
    return_cost = distance_matrix[:, 0]  # Shape (n,)
    
    # Total cost of adding node j to the path from i and then returning to 0
    # This represents the "budget" consumed by this move and the subsequent return.
    total_cost = distance_matrix + return_cost[None, :]
    
    # Alpha controls the steepness of the decay.
    # With max_len=3 and unit square, typical costs are < 2.
    # An alpha of 2.0 makes exp(-2*1) ~ 0.13, providing strong discrimination.
    alpha = 2.0
    
    # Exponential decay
    # We multiply by prize to ensure high-value nodes are preferred.
    # If prize is 0, heuristic is 0 (which will be clipped to 1e-9).
    heuristic = prizes[None, :] * np.exp(-alpha * total_cost)
    
    # Handle any non-finite values (though exp is stable, good practice)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    # Ensure positive values as per contract
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
