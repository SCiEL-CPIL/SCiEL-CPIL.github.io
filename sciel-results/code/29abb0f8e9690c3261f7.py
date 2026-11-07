"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    Larger values = more attractive edges.
    
    Incorporates:
    - Distance-based attractiveness (shorter = more attractive, squared decay)
    - Demand-awareness: customers with lower demand are more attractive
      (easier to fit into remaining capacity)
    - Normalization for numerical stability
    """
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    demands = np.asarray(demands, dtype=np.float64)
    
    n = distance_matrix.shape[0]
    
    # Base distance attractiveness: 1/d^2 (sharper decay than 1/d)
    # Add small epsilon to avoid division by zero (diagonal)
    d = distance_matrix + 1e-9
    dist_attr = 1.0 / (d * d)
    
    # Demand-awareness: penalize edges to high-demand customers
    # For each edge i->j, if j is a customer (j > 0), scale by (1 - demand_j / capacity)
    # This makes low-demand customers more attractive (easier to pack)
    demand_penalty = np.ones((n, n), dtype=np.float64)
    for j in range(1, n):  # skip depot
        # Scale factor: lower demand = higher attractiveness
        # Use (capacity - demand_j) / capacity as a simple penalty
        demand_penalty[:, j] = (capacity - demands[j]) / capacity
    
    # Combine: distance attractiveness * demand penalty
    heuristic = dist_attr * demand_penalty
    
    # Ensure non-negative and replace non-positive with small value
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    # Normalize: scale so that the max is around 1.0 for stability
    max_val = np.max(heuristic)
    if max_val > 0:
        heuristic = heuristic / max_val
    
    return heuristic
# EVOLVE-BLOCK-END
