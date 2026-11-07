"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def heuristics(distance_matrix):
    """
    Compute an edge-attractiveness heuristic for ACO on the TSP.
    
    Strategy: Multi-source consensus using MST edges, nearest-neighbor edges,
    and inverse-distance. This provides a robust structural guide while maintaining
    continuous variation to avoid over-fitting to a single backbone.
    """
    n = distance_matrix.shape[0]
    if n < 2:
        return np.ones_like(distance_matrix)
    
    # Ensure symmetry and handle self-loops
    dm = (distance_matrix + distance_matrix.T) / 2.0
    np.fill_diagonal(dm, 0.0)
    
    # 1. Minimum Spanning Tree edges (structural backbone)
    # Use Prim's algorithm for MST on the distance matrix
    in_mst = np.zeros(n, dtype=bool)
    min_dist = np.full(n, np.inf)
    min_dist[0] = 0.0
    mst_edges = np.zeros((n, n), dtype=float)
    
    for _ in range(n):
        # Find unvisited node with minimum distance
        unvisited = ~in_mst
        if not np.any(unvisited):
            break
        # Set visited nodes to inf for selection
        tmp = min_dist.copy()
        tmp[in_mst] = np.inf
        u = int(np.argmin(tmp))
        
        if not np.isfinite(min_dist[u]):
            break
            
        in_mst[u] = True
        
        # Update distances
        for v in range(n):
            if not in_mst[v] and dm[u, v] < min_dist[v]:
                min_dist[v] = dm[u, v]
        
        # Mark MST edge (connect to the node that gave us the min_dist)
        # We need to track the parent
        # Actually, let's just record all edges that were part of the MST construction
        # But a simpler way: after building MST, we can identify edges.
        # Let's use a different approach: Kruskal's or just track parents.
        
    # Let's redo MST properly to get edges
    # Prim's with parent tracking
    in_mst = np.zeros(n, dtype=bool)
    min_dist = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=int)
    min_dist[0] = 0.0
    
    for _ in range(n):
        unvisited = ~in_mst
        if not np.any(unvisited):
            break
        tmp = min_dist.copy()
        tmp[in_mst] = np.inf
        u = int(np.argmin(tmp))
        
        if not np.isfinite(min_dist[u]):
            break
            
        in_mst[u] = True
        
        if parent[u] != -1:
            mst_edges[u, parent[u]] = 1.0
            mst_edges[parent[u], u] = 1.0
            
        for v in range(n):
            if not in_mst[v] and dm[u, v] < min_dist[v]:
                min_dist[v] = dm[u, v]
                parent[v] = u
    
    # 2. Nearest Neighbor edges (local structure)
    nn_edges = np.zeros((n, n), dtype=float)
    nn_dist = np.full((n, n), np.inf)
    for i in range(n):
        # Find 3 nearest neighbors for each node (excluding self)
        dists = dm[i].copy()
        dists[i] = np.inf
        # Get indices of 3 smallest
        k = min(3, n - 1)
        idx = np.argpartition(dists, k)[:k]
        for j in idx:
            nn_edges[i, j] = 1.0
            nn_edges[j, i] = 1.0  # Symmetric
            nn_dist[i, j] = dm[i, j]
    
    # 3. Inverse distance (continuous baseline)
    # Avoid division by zero
    inv_dist = 1.0 / (dm + 1e-9)
    np.fill_diagonal(inv_dist, 0.0)
    
    # Normalize inv_dist to [0, 1] range for combination
    max_inv = np.max(inv_dist)
    if max_inv > 0:
        inv_dist_norm = inv_dist / max_inv
    else:
        inv_dist_norm = np.zeros_like(inv_dist)
    
    # Combine heuristics:
    # - MST edges get a high boost
    # - NN edges get a medium boost
    # - Base is inverse distance
    # We want larger values to be more attractive.
    
    # Strategy: 
    # heuristic = base + alpha * mst + beta * nn
    # But we need to ensure non-negative and positive.
    # Let's use a multiplicative or additive scheme.
    
    # Let's try: heuristic = inv_dist_norm * (1 + 2.0 * mst_edges + 1.0 * nn_edges)
    # This boosts MST edges by 3x and NN edges by 2x relative to base.
    
    heuristic = inv_dist_norm * (1.0 + 2.0 * mst_edges + 1.0 * nn_edges)
    
    # Ensure diagonal is 0 (or small) and values are positive
    np.fill_diagonal(heuristic, 0.0)
    
    # Handle any non-positive values
    heuristic[heuristic <= 0] = 1e-9
    
    # Check for non-finite values
    if not np.all(np.isfinite(heuristic)):
        # Fallback to simple inverse distance
        heuristic = 1.0 / (dm + 1e-9)
        np.fill_diagonal(heuristic, 0.0)
        heuristic[heuristic <= 0] = 1e-9
    
    return heuristic

# EVOLVE-BLOCK-END
