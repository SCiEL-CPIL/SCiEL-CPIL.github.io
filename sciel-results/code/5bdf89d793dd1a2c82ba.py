"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    
    The heuristic is designed to:
    1. Prefer shorter edges (inverse distance)
    2. Account for demand: edges to high-demand nodes are slightly less attractive
       unless they are very short (to encourage clustering of similar demand)
    3. Use a soft-min surrogate to smooth the interaction between distance and demand
    
    Args:
        distance_matrix: (n, n) array of pairwise distances
        coordinates: (n, 2) array of coordinates
        demands: (n,) array of demands (node 0 is depot with demand 0)
        capacity: vehicle capacity
    
    Returns:
        (n, n) edge-attractiveness matrix (larger is more attractive)
    """
    n = len(demands)
    dm = np.asarray(distance_matrix, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    
    # Inverse distance as base attractiveness
    # Add small epsilon to avoid division by zero
    inv_dist = 1.0 / (dm + 1e-9)
    
    # Demand-based adjustment:
    # We want to encourage visiting nodes that can be served together efficiently.
    # A simple heuristic: scale by the square root of (capacity / demand_j) for j > 0
    # This makes high-demand nodes less attractive per edge, but since they must be visited,
    # the ant will still visit them when forced.
    
    # Create a demand weight matrix: w[i, j] = sqrt(capacity / max(dem[j], 1))
    # For the depot (j=0), demand is 0, so we treat it specially
    dem_safe = np.maximum(dem, 1.0)
    dem_safe[0] = capacity  # Depot has no demand, treat as full capacity
    
    # Demand efficiency: higher for low-demand nodes
    demand_eff = np.sqrt(capacity / dem_safe)
    
    # Scale the inverse distance by demand efficiency of the destination
    # heuristic[i, j] = inv_dist[i, j] * demand_eff[j]
    heuristic = inv_dist * demand_eff[np.newaxis, :]
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
