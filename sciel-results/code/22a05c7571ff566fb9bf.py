"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization of VRP.
    
    This heuristic combines:
    1. Inverse distance (standard)
    2. Spatial clustering via grid-based cell assignment to encourage compact routes
    3. Demand-aware adjustment to prefer customers that fit well in remaining capacity
    
    Returns a matrix where larger values are more attractive.
    """
    n = distance_matrix.shape[0]
    
    # Base: inverse distance
    inv_dist = 1.0 / (distance_matrix + 1e-9)
    
    # Grid-based spatial clustering
    # Divide the unit square into a grid and boost intra-cell edges
    # This encourages ants to form compact routes within regions
    n_cells = 4  # 4x4 grid
    cell_size = 1.0 / n_cells
    
    # Assign each node to a cell
    cells = np.zeros(n, dtype=int)
    for i in range(n):
        cx = int(coordinates[i, 0] / cell_size)
        cy = int(coordinates[i, 1] / cell_size)
        cx = min(cx, n_cells - 1)
        cy = min(cy, n_cells - 1)
        cells[i] = cy * n_cells + cx
    
    # Build cell affinity matrix: 1 if same cell, 0 otherwise
    cell_affinity = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if cells[i] == cells[j]:
                cell_affinity[i, j] = 1.0
    
    # Boost intra-cell edges and slightly penalize inter-cell edges
    # Use a multiplicative factor: intra-cell edges get boosted, inter-cell get reduced
    intra_boost = 1.5
    inter_penalty = 0.8
    
    spatial_factor = np.where(cell_affinity > 0.5, intra_boost, inter_penalty)
    
    # Demand-aware adjustment: prefer edges to nodes with moderate demand
    # Nodes with very high demand are harder to fit, so slightly reduce their attractiveness
    # But don't reduce too much, as we still need to visit them
    demand_factor = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                # Reduce attractiveness of edges to high-demand nodes slightly
                # Scale by how well the demand fits in capacity
                d_j = demands[j]
                if d_j > 0:
                    # Higher demand -> slightly lower attractiveness
                    demand_factor[i, j] = 1.0 / (1.0 + 0.1 * d_j / capacity)
    
    # Combine all factors
    heuristic = inv_dist * spatial_factor * demand_factor
    
    # Ensure all values are positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
