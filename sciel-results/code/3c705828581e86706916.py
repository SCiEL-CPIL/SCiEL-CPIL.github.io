"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute edge-attractiveness matrix for ant colony optimization of CVRP.
    Uses inverse distance with demand-aware adjustment to guide ants toward
    efficient routing that respects vehicle capacity constraints.
    """
    n = distance_matrix.shape[0]
    dm = np.asarray(distance_matrix, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    
    # Base heuristic: inverse distance (closer nodes are more attractive)
    base_heuristic = 1.0 / (dm + 1e-9)
    
    # Demand-aware adjustment: penalize edges to high-demand nodes
    # This helps the ACO find solutions that group high-demand customers
    # more efficiently, reducing the number of trips needed
    # Using a stronger penalty than before to better guide capacity-aware routing
    alpha = 0.15
    demand_penalty = 1.0 / (1.0 + alpha * dem)
    
    # Apply demand penalty to destination nodes
    adjusted_heuristic = base_heuristic * demand_penalty[np.newaxis, :]
    
    # Ensure all values are positive and finite
    adjusted_heuristic[~np.isfinite(adjusted_heuristic)] = 1e-9
    adjusted_heuristic[adjusted_heuristic <= 0] = 1e-9
    
    return adjusted_heuristic
# EVOLVE-BLOCK-END
