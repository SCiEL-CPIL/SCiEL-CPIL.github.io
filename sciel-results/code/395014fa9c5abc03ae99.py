# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    """
    Compute an edge-attractiveness heuristic matrix for ant colony optimization
    of the TSP. Uses a combination of inverse distance and a local density
    adjustment to guide ants towards promising edges.
    
    The heuristic is designed to be larger for shorter edges (more attractive),
    with an additional boost for edges that connect points in less dense regions,
    which can help avoid getting trapped in local optima.
    """
    # Ensure we're working with a numpy array
    dm = np.asarray(distance_matrix, dtype=np.float64)
    
    # Basic inverse distance heuristic: shorter edges are more attractive
    # Add small epsilon to avoid division by zero
    base_heuristic = 1.0 / (dm + 1e-9)
    
    # Compute local density for each node
    # For each node, compute the mean distance to its k-nearest neighbors
    n = dm.shape[0]
    
    # Use a small k for local density estimation
    k = min(5, n - 1)
    if k < 1:
        return base_heuristic
    
    # For each node, find the k-nearest neighbors and compute mean distance
    local_density = np.zeros(n)
    for i in range(n):
        # Get distances from node i to all other nodes
        dists = dm[i].copy()
        dists[i] = np.inf  # Exclude self
        # Get the k smallest distances
        if k < n - 1:
            k_nearest = np.partition(dists, k)[:k]
        else:
            k_nearest = dists
        local_density[i] = np.mean(k_nearest)
    
    # Normalize local density to [0, 1] range
    # Lower density means the node is more isolated
    min_density = np.min(local_density)
    max_density = np.max(local_density)
    if max_density - min_density > 1e-12:
        normalized_density = (local_density - min_density) / (max_density - min_density)
    else:
        normalized_density = np.zeros(n)
    
    # Create a density-based boost matrix
    # Edges between less dense nodes get a slight boost
    # This helps ants explore less crowded regions
    density_boost = 1.0 + 0.5 * (1.0 - normalized_density[:, None]) * (1.0 - normalized_density[None, :])
    
    # Combine base heuristic with density boost
    heuristic = base_heuristic * density_boost
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
