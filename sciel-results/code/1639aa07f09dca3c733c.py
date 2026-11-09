import numpy as np
import time

def heuristics(distance_matrix):
    """
    Heuristic matrix for ant colony optimization of TSP.
    
    Uses a softened inverse distance (d^-0.8) as the base heuristic,
    combined with a symmetric k-nearest-neighbor boost and a small
    minimum spanning tree boost to guide ants toward natural tour edges.
    
    The softened inverse reduces the dominance of the nearest neighbor,
    allowing ants to explore more broadly and escape local optima.
    The k-NN boost encourages ants to choose edges that are part of
    the local neighborhood structure.
    The MST boost provides a global structural prior.
    """
    start_time = time.time()
    
    dm = np.asarray(distance_matrix, dtype=float)
    n = dm.shape[0]
    
    # Check for non-finite values
    if not np.all(np.isfinite(dm)):
        raise ValueError("Distance matrix contains non-finite values")
    
    # Base heuristic: softened inverse distance
    alpha = 0.8
    d_safe = dm + 1e-9
    base_heuristic = 1.0 / (d_safe ** alpha)
    
    # Symmetric k-NN boost
    k = 4
    knn_boost = np.ones((n, n))
    for i in range(n):
        dists = dm[i].copy()
        dists[i] = np.inf
        knn_idx = np.argsort(dists)[:k]
        for j in knn_idx:
            knn_boost[i, j] = 1.5
            knn_boost[j, i] = 1.5
    
    # MST boost (small)
    in_mst = np.zeros(n, dtype=bool)
    min_dist = np.full(n, np.inf)
    min_dist[0] = 0
    mst_edge = np.zeros((n, n))
    
    for _ in range(n):
        u = -1
        for i in range(n):
            if not in_mst[i] and (u == -1 or min_dist[i] < min_dist[u]):
                u = i
        in_mst[u] = True
        for v in range(n):
            if not in_mst[v] and dm[u, v] < min_dist[v]:
                min_dist[v] = dm[u, v]
                mst_edge[u, v] = 1
                mst_edge[v, u] = 1
    
    mst_boost = 1.0 + 0.3 * mst_edge
    
    # Combine boosts
    heuristic = base_heuristic * knn_boost * mst_boost
    
    # Ensure diagonal is very small
    np.fill_diagonal(heuristic, 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Normalize so the mean is around 1
    mean_val = heuristic[heuristic > 0].mean()
    if mean_val > 0:
        heuristic = heuristic / mean_val
    
    return heuristic
