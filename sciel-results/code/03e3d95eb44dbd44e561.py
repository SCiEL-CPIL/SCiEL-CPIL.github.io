"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization of CVRP.
    
    Combines:
    1. Inverse distance (base attractiveness)
    2. Spatial coherence via grid-based cell boosting
    3. Directional consistency (smooth turns)
    """
    n = distance_matrix.shape[0]
    coords = np.asarray(coordinates, dtype=np.float64)
    dist = np.asarray(distance_matrix, dtype=np.float64)
    
    # Base: inverse distance
    inv_dist = 1.0 / (dist + 1e-9)
    
    # Grid-based spatial coherence
    # Divide unit square into cells; boost edges within same/adjacent cells
    n_cells = 8  # 8x8 grid
    cell_size = 1.0 / n_cells
    
    # Assign each node to a cell
    cell_ids = np.zeros(n, dtype=np.int64)
    for i in range(n):
        cx = int(min(max(coords[i, 0], 0.0), 0.999999) / cell_size)
        cy = int(min(max(coords[i, 1], 0.0), 0.999999) / cell_size)
        cell_ids[i] = cy * n_cells + cx
    
    # Cell distance matrix (Manhattan distance in cell space)
    cell_mat = np.abs(cell_ids[:, None] // n_cells - cell_ids[None, :] // n_cells) + \
               np.abs(cell_ids[:, None] % n_cells - cell_ids[None, :] % n_cells)
    
    # Spatial boost: edges in same cell get highest boost, adjacent cells lower
    # Use exponential decay with cell distance
    spatial_boost = np.exp(-0.5 * cell_mat)
    
    # Directional consistency: for each (i,j), check if the direction from depot to i
    # aligns with direction from i to j. Prefer small angles (smooth routing).
    # This encourages ants to sweep around the depot rather than zigzag.
    depot = coords[0]
    # Vector from depot to each node
    v_depot_to_node = coords - depot[None, :]
    # Normalize
    norms = np.linalg.norm(v_depot_to_node, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    v_depot_to_node_norm = v_depot_to_node / norms
    
    # For each pair (i, j), compute angle between (depot->i) and (i->j)
    # Vector from i to j
    v_i_to_j = coords[None, :, :] - coords[:, None, :]  # shape (n, n, 2)
    v_i_to_j_norm = np.linalg.norm(v_i_to_j, axis=2, keepdims=True)
    v_i_to_j_norm[v_i_to_j_norm == 0] = 1.0
    v_i_to_j_unit = v_i_to_j / v_i_to_j_norm
    
    # Cosine of angle between depot->i and i->j
    # v_depot_to_node_norm[i] dot v_i_to_j[i, j]
    cos_angle = np.sum(v_depot_to_node_norm[:, None, :] * v_i_to_j, axis=2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Prefer small angles: use (1 + cos_angle) / 2 as a smooth weight in [0, 1]
    # But we want to prefer going "forward" (small angle) over "back" (large angle)
    directional_weight = (1.0 + cos_angle) / 2.0
    
    # Combine: base * spatial_boost * directional_weight
    # Weight the components
    heuristic = inv_dist * spatial_boost * (0.5 + 0.5 * directional_weight)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
