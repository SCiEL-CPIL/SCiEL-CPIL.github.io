"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def _compute_mst_mask(distance_matrix):
    """
    Computes the adjacency mask of the Minimum Spanning Tree (MST)
    for a complete graph given its distance matrix.
    Uses Prim's algorithm starting from node 0.
    """
    n = distance_matrix.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=bool)
    
    in_mst = np.zeros(n, dtype=bool)
    min_dist = np.full(n, np.inf)
    min_dist[0] = 0.0
    parent = np.full(n, -1, dtype=int)
    
    for _ in range(n):
        # Find the unvisited node with the smallest connection weight
        # O(N^2) is fine for N=50
        u = -1
        best_dist = np.inf
        for i in range(n):
            if not in_mst[i] and min_dist[i] < best_dist:
                best_dist = min_dist[i]
                u = i
        
        if u == -1:
            break
            
        in_mst[u] = True
        
        # Update distances to neighbors
        for v in range(n):
            if not in_mst[v] and distance_matrix[u, v] < min_dist[v]:
                min_dist[v] = distance_matrix[u, v]
                parent[v] = u
                
    # Construct the symmetric MST mask
    mst_mask = np.zeros((n, n), dtype=bool)
    for v in range(1, n):
        if parent[v] != -1:
            mst_mask[parent[v], v] = True
            mst_mask[v, parent[v]] = True
            
    return mst_mask

def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for ACO.
    Uses a heuristic that boosts edges belonging to the Minimum Spanning Tree (MST)
    to encourage connectivity and local geometric structure, while maintaining
    inverse-distance scaling for all other edges.
    """
    D = np.asarray(distance_matrix, dtype=float)
    n = D.shape[0]
    
    # Base heuristic: inverse distance
    # Add epsilon to avoid division by zero for diagonal
    epsilon = 1e-9
    H_base = 1.0 / (D + epsilon)
    
    # Compute MST mask
    mst_mask = _compute_mst_mask(D)
    
    # Apply boost to MST edges
    # Factor of 3.0 means MST edges are 3x more attractive than non-MST edges of same distance
    boost_factor = 3.0
    
    H = H_base.copy()
    H[mst_mask] *= boost_factor
    
    # Ensure diagonal is handled (though ACO ignores visited nodes, 
    # keeping it 0 or small is safer for any generic use)
    np.fill_diagonal(H, 0.0)
    
    # Replace non-positive or non-finite values with 1e-9 as per contract
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
