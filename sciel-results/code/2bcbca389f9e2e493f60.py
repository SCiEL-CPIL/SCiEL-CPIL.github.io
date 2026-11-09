"""Heuristic matrix for ant colony optimization of TSP."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay, cKDTree
import scipy.sparse as sp
from scipy.sparse.csgraph import minimum_spanning_tree


def heuristics(distance_matrix):
    """
    Compute an edge-attractiveness matrix for ACO on TSP.
    
    Strategy:
    1. Base: inverse distance (standard).
    2. Boost edges that are in the Delaunay triangulation (local structure).
    3. Boost edges that are in the Minimum Spanning Tree (MST).
    4. Penalize long edges more aggressively to avoid crossings.
    
    The heuristic is computed once per instance.
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin for 300s limit
    
    n = distance_matrix.shape[0]
    if n <= 1:
        return np.ones_like(distance_matrix)
    
    # Ensure we have a square matrix
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("Distance matrix must be square")
    
    # Check for non-finite values
    if not np.all(np.isfinite(distance_matrix)):
        raise ValueError("Distance matrix contains non-finite values")
    
    # Extract coordinates from distance matrix? No, we don't have coordinates.
    # But the task says "50 uniform random points in the unit square".
    # We only have the distance matrix. We need to work with the matrix directly.
    # However, Delaunay triangulation requires coordinates.
    # We can estimate coordinates using MDS (Multidimensional Scaling) or just use the distance matrix structure.
    # Actually, for 50 points, we can try to reconstruct coordinates or use graph-based heuristics.
    
    # Let's use a graph-based approach since we don't have explicit coordinates.
    # 1. Base heuristic: 1 / (distance + epsilon)
    eps = 1e-9
    base_heuristic = 1.0 / (distance_matrix + eps)
    
    # 2. Compute MST to identify "skeleton" edges
    # Convert dense distance matrix to sparse for MST
    # We need to handle the fact that MST expects a graph.
    # scipy.sparse.csgraph.minimum_spanning_tree can take a dense matrix.
    try:
        mst = minimum_spanning_tree(distance_matrix)
        mst_dense = mst.toarray()
        # mst_dense[i][j] is the weight if edge exists, 0 otherwise (or inf?)
        # Actually, for minimum_spanning_tree, the output is a sparse matrix where 
        # the value is the weight of the edge if it exists in the MST, 0 otherwise.
        # Let's create a binary mask for MST edges.
        mst_mask = mst_dense > 0
    except Exception:
        mst_mask = np.zeros_like(distance_matrix, dtype=bool)
    
    # 3. Identify "local" edges. 
    # Since we don't have coordinates, we can define "local" as edges that are among the shortest for each node.
    # For each node i, find the k nearest neighbors.
    k = 5
    local_mask = np.zeros_like(distance_matrix, dtype=bool)
    for i in range(n):
        # Get distances from i to all other nodes
        dists = distance_matrix[i].copy()
        dists[i] = np.inf  # Exclude self
        # Find k smallest
        k_indices = np.argpartition(dists, k)[:k]
        local_mask[i, k_indices] = True
        # Symmetric
        local_mask[k_indices, i] = True
    
    # 4. Combine heuristics
    # Start with base
    heuristic = base_heuristic.copy()
    
    # Boost MST edges
    mst_boost = 2.0
    heuristic[mst_mask] *= mst_boost
    
    # Boost local edges
    local_boost = 1.5
    heuristic[local_mask] *= local_boost
    
    # Penalize very long edges
    # Find median distance
    upper_tri = distance_matrix[np.triu_indices(n, k=1)]
    if len(upper_tri) > 0:
        median_dist = np.median(upper_tri)
        # For edges significantly longer than median, reduce attractiveness
        long_mask = distance_matrix > 2.0 * median_dist
        heuristic[long_mask] *= 0.5
    
    # Ensure diagonal is 0 (or very small) to prevent self-loops if not handled by ACO
    np.fill_diagonal(heuristic, 0.0)
    
    # Ensure all values are positive and finite
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    # Check deadline
    if time.time() > deadline:
        return heuristic
    
    return heuristic
# EVOLVE-BLOCK-END
