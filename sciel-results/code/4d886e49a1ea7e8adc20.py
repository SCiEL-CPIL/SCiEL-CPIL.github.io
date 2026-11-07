"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    """
    Spectral graph-theoretic heuristic combining MST structure with distance-based
    attractiveness. Edges in the MST get boosted based on node degrees, modeling
    how strongly an edge connects high-degree hubs in the tour's backbone.
    """
    n = distance_matrix.shape[0]
    d = distance_matrix.copy()
    np.fill_diagonal(d, np.inf)
    
    # Compute MST using Prim's algorithm (O(n^2) for dense graph)
    in_mst = [False] * n
    min_dist = [np.inf] * n
    min_dist[0] = 0
    mst_edges = []
    
    for _ in range(n):
        # Find the unvisited node with minimum distance
        u = -1
        best = np.inf
        for i in range(n):
            if not in_mst[i] and min_dist[i] < best:
                best = min_dist[i]
                u = i
        
        if u == -1:
            break
        
        in_mst[u] = True
        
        # Update distances to other nodes
        for v in range(n):
            if not in_mst[v] and d[u, v] < min_dist[v]:
                min_dist[v] = d[u, v]
                mst_edges.append((u, v))
    
    # Build MST adjacency and compute degrees
    mst_adj = np.zeros((n, n), dtype=np.float64)
    mst_degree = np.zeros(n, dtype=np.float64)
    for u, v in mst_edges:
        if u != v:
            mst_adj[u, v] = 1.0
            mst_adj[v, u] = 1.0
            mst_degree[u] += 1.0
            mst_degree[v] += 1.0
    
    # Compute rank-based heuristic (same as before)
    ranks = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        sorted_indices = np.argsort(d[i])
        for rank_idx, j in enumerate(sorted_indices):
            ranks[i][j] = rank_idx + 1.0
    
    # Base heuristic: inverse distance with smoothing
    eta_base = 1.0 / (d + 1e-9)
    np.fill_diagonal(eta_base, 0.0)
    
    # MST boost: edges in MST get multiplied by a factor based on node degrees
    # Higher degree nodes are more "central" to the tour backbone
    mst_boost = 1.0 + 0.5 * (mst_degree[:, None] + mst_degree[None, :]) * mst_adj
    
    # Combine: base heuristic modulated by MST structure
    eta = eta_base * mst_boost
    
    # Also incorporate rank-based component for local preference
    eta_rank = 1.0 / (ranks + 1.0)
    np.fill_diagonal(eta_rank, 0.0)
    
    # Final heuristic: blend distance-based and rank-based with MST boost
    eta_final = 0.7 * eta + 0.3 * eta_rank
    np.fill_diagonal(eta_final, 0.0)
    
    # Ensure all values are positive (as required by ACO)
    eta_final = np.maximum(eta_final, 1e-9)
    np.fill_diagonal(eta_final, 1e-9)
    
    return eta_final
# EVOLVE-BLOCK-END
