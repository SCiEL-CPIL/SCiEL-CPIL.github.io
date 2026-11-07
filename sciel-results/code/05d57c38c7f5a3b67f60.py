"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes a spatially-smoothed, demand-aware heuristic matrix for ACO.
    
    The heuristic combines:
    1. Inverse distance (base attractiveness)
    2. Demand penalty (lower demand = more attractive)
    3. Spatial coherence bonus (edges that align with local neighborhood structure)
    
    This encourages ants to construct routes that are both short and load-balanced.
    
    Args:
        distance_matrix: NxN matrix of Euclidean distances.
        coordinates: Nx2 array of coordinates.
        demands: Array of length N with customer demands (floats).
        capacity: Vehicle capacity.
        
    Returns:
        NxN matrix of edge attractiveness (larger is more attractive).
    """
    d_mat = np.asarray(distance_matrix, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    n = d_mat.shape[0]
    
    epsilon = 1e-9
    
    # 1. Base: inverse distance
    base = 1.0 / (d_mat + epsilon)
    
    # 2. Demand penalty: prefer low-demand destinations
    # Use a softer penalty to avoid over-suppressing high-demand nodes
    demand_factor = 1.0 / (1.0 + 0.3 * dem)
    heuristic = base * demand_factor[np.newaxis, :]
    
    # 3. Spatial coherence: for each node i, compute local neighborhood centroid
    # and boost edges from i to nodes that are in the same "cluster direction"
    # This is approximated by checking if the angle between (coords[j]-coords[i])
    # and the local principal direction is small.
    
    # Compute k-nearest neighbors for each node (excluding self)
    k = 5
    # For each i, find k nearest neighbors
    nn_indices = np.zeros((n, k), dtype=np.int64)
    for i in range(n):
        dists = d_mat[i].copy()
        dists[i] = np.inf
        nn_indices[i] = np.argpartition(dists, k)[:k]
    
    # For each node i, compute the mean direction to its k nearest neighbors
    mean_dirs = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        nn = nn_indices[i]
        vecs = coords[nn] - coords[i]
        # Normalize and average
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms < epsilon] = epsilon
        vecs_norm = vecs / norms
        mean_dirs[i] = np.mean(vecs_norm, axis=0)
    
    # Normalize mean directions
    md_norms = np.linalg.norm(mean_dirs, axis=1, keepdims=True)
    md_norms[md_norms < epsilon] = epsilon
    mean_dirs_norm = mean_dirs / md_norms
    
    # For each edge (i, j), compute alignment with mean direction of i
    # alignment[i, j] = dot(mean_dirs_norm[i], (coords[j]-coords[i])/||coords[j]-coords[i]||)
    edge_vecs = coords[np.newaxis, :, :] - coords[:, np.newaxis, :]  # (n, n, 2)
    edge_norms = np.linalg.norm(edge_vecs, axis=2)  # (n, n)
    edge_norms_safe = np.where(edge_norms > epsilon, edge_norms, epsilon)
    edge_dirs = edge_vecs / edge_norms_safe[:, :, np.newaxis]  # (n, n, 2)
    
    # Alignment: dot product of mean_dirs_norm[i] with edge_dirs[i, j]
    # mean_dirs_norm[i] is (2,), edge_dirs[i, j] is (2,)
    alignment = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        alignment[i, :] = np.sum(mean_dirs_norm[i] * edge_dirs[i, :], axis=1)
    
    # Boost edges with positive alignment (in the direction of local flow)
    # Use a mild boost: 1 + beta * max(0, alignment)
    beta = 0.5
    spatial_boost = 1.0 + beta * np.maximum(0.0, alignment)
    
    heuristic = heuristic * spatial_boost
    
    # 4. Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
