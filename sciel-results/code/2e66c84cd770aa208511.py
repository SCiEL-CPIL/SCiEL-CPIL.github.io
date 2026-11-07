"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix):
    """
    Compute heuristic matrix for ACO on TSP.
    Uses tuned inverse distance with MST and k-NN structural guidance.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    n = distance_matrix.shape[0]
    dm = distance_matrix + 1e-9
    
    # 1. Base heuristic: inverse distance with exponent 1.5
    heuristic = 1.0 / (dm ** 1.5)
    
    # 2. Compute Minimum Spanning Tree (MST) for structural guidance
    in_mst = [False] * n
    min_dist = [float('inf')] * n
    min_dist[0] = 0.0
    parent = [-1] * n
    
    for _ in range(n):
        u = -1
        best = float('inf')
        for v in range(n):
            if not in_mst[v] and min_dist[v] < best:
                best = min_dist[v]
                u = v
        
        if u == -1:
            break
        
        in_mst[u] = True
        
        for v in range(n):
            if not in_mst[v] and distance_matrix[u, v] < min_dist[v]:
                min_dist[v] = distance_matrix[u, v]
                parent[v] = u
    
    # 3. Build MST edge set
    mst_edges = set()
    for v in range(1, n):
        if parent[v] != -1:
            u, v_min = parent[v], v
            mst_edges.add((min(u, v_min), max(u, v_min)))
    
    # 4. Boost MST edges with moderate factor
    mst_boost = 2.0
    for (u, v) in mst_edges:
        heuristic[u, v] *= mst_boost
        heuristic[v, u] *= mst_boost
    
    # 5. Add k-nearest neighbor boost for broader guidance
    k = 6
    for i in range(n):
        sorted_nodes = np.argsort(distance_matrix[i])
        for j in sorted_nodes[:k]:
            if j != i:
                heuristic[i, j] *= 1.15
                heuristic[j, i] *= 1.15
    
    # 6. Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
