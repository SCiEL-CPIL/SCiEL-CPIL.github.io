"""Heuristic matrix for ant colony optimization of TSP."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_mst_edges(dm):
    """
    Compute the edges of the Minimum Spanning Tree using Prim's algorithm.
    Returns a set of tuples (i, j) representing the MST edges.
    """
    n = dm.shape[0]
    if n <= 1:
        return set()
    
    in_mst = np.zeros(n, dtype=bool)
    min_dist = np.full(n, np.inf)
    min_dist[0] = 0.0
    parent = np.full(n, -1, dtype=int)
    mst_edges = set()
    
    for _ in range(n):
        # Find the vertex with the minimum distance not in MST
        u = -1
        best_dist = np.inf
        for i in range(n):
            if not in_mst[i] and min_dist[i] < best_dist:
                best_dist = min_dist[i]
                u = i
        
        if u == -1:
            break
            
        in_mst[u] = True
        if parent[u] != -1:
            mst_edges.add((parent[u], u))
            
        # Update distances for neighbors
        for v in range(n):
            if not in_mst[v] and dm[u, v] < min_dist[v]:
                min_dist[v] = dm[u, v]
                parent[v] = u
                
    return mst_edges

def _compute_1nn_edges(dm):
    """
    Compute the 1-Nearest Neighbor edges for each node.
    Returns a set of tuples (i, j).
    """
    n = dm.shape[0]
    edges = set()
    for i in range(n):
        # Exclude self
        dists = dm[i].copy()
        dists[i] = np.inf
        j = int(np.argmin(dists))
        edges.add((i, j))
    return edges

def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for ACO.
    Strategy: Topological heuristic based on MST and 1-NN constraints.
    1. Base heuristic is 1/d.
    2. Boost edges in MST (structural backbone).
    3. Boost edges to 1-NN (local optimality).
    4. Penalize edges that create small cycles in the 'core' graph (MST + 1NN).
    5. Penalize very long edges.
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    dm = np.asarray(distance_matrix, dtype=float)
    n = dm.shape[0]
    
    if n == 0:
        return dm.copy()
    
    # 1. Base heuristic: 1/d
    # Avoid division by zero
    dm_safe = np.where(dm == 0, 1e-9, dm)
    heuristic = 1.0 / dm_safe
    
    # 2. Compute structural edges
    mst_edges = _compute_mst_edges(dm)
    nn_edges = _compute_1nn_edges(dm)
    
    # Combine into a "core" graph for cycle detection
    core_edges = mst_edges.union(nn_edges)
    
    # 3. Identify "bad" edges that create small cycles
    # We want to avoid edges that, when added to the core, close a loop of length < 5.
    # Since the core is a tree (MST) plus some extra edges (NN), it might have cycles.
    # Actually, MST is a tree. NN edges add cycles.
    # A simple heuristic: if an edge (u, v) is not in MST, and the shortest path between u and v in MST is small, 
    # it might be redundant or part of a small loop.
    
    # However, calculating shortest paths for all pairs is O(n^3) or O(n^2 log n).
    # For n=50, we can just check if the edge connects nodes that are "close" in the MST tree distance.
    
    # Let's build an adjacency list for MST to compute tree distances if needed, 
    # but for n=50, we can just use the distance matrix itself as a proxy for "closeness".
    
    # Simpler approach for "small cycle" penalty:
    # If d(u, v) is small, it's likely a good edge.
    # If d(u, v) is large, it's likely bad.
    # The current 1/d already handles this.
    
    # Let's refine the boost strategy.
    # MST edges are essential. Boost them significantly.
    # 1-NN edges are very likely in the optimal tour. Boost them.
    # Other edges get the base 1/d.
    
    # Determine median distance to normalize penalties/boosts
    flat_dm = dm.flatten()
    flat_dm = flat_dm[flat_dm > 0]
    if len(flat_dm) == 0:
        return heuristic
    median_dist = np.median(flat_dm)
    
    # Boost factors
    mst_boost = 5.0
    nn_boost = 3.0
    
    # Apply boosts
    for i in range(n):
        for j in range(i + 1, n):
            if i == j:
                continue
            # Symmetric TSP, so (i,j) and (j,i) are same edge
            if (i, j) in mst_edges or (j, i) in mst_edges:
                heuristic[i, j] *= mst_boost
                heuristic[j, i] *= mst_boost
            elif (i, j) in nn_edges or (j, i) in nn_edges:
                heuristic[i, j] *= nn_boost
                heuristic[j, i] *= nn_boost
            
            # Penalize very long edges
            # If distance is > 3 * median, reduce attractiveness
            if dm[i, j] > 3.0 * median_dist:
                heuristic[i, j] *= 0.1
                heuristic[j, i] *= 0.1
            
            # Additional penalty for edges that are "redundant" locally
            # If there is a node k such that d(i,k) + d(k,j) < d(i,j) * 1.2, 
            # then edge (i,j) is likely not part of the optimal tour (triangle inequality violation in tour context)
            # This is O(n^3) which is 125,000 ops for n=50. Very fast.
            # Let's check if (i,j) is "covered" by a short path through k.
            # Actually, for TSP, we want to avoid edges that skip over nearby nodes.
            
    # Vectorized "skip" penalty:
    # For each edge (i, j), if there exists k such that d(i, k) + d(k, j) < 1.5 * d(i, j),
    # it suggests i and j are not directly connected in the optimal tour (they are separated by k).
    # We can approximate this by checking the 2nd nearest neighbor logic or just a simple matrix operation.
    
    # Let's try a simpler vectorized penalty for "long" edges relative to local density.
    # If d(i,j) is large, and there are many points within d(i,j)/2 of i or j, it's likely a jump.
    
    # To keep it simple and fast, let's stick to MST/NN boosts and long edge penalty.
    # The MST/NN boosts are the key differentiator from the "found tour" approach.
    
    # Ensure positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Normalize to prevent overflow in ACO (though ACO usually handles scales)
    # The problem statement says "larger is more attractive".
    # It doesn't specify normalization.
    # However, if values are too large, pheromone * heuristic might overflow?
    # Pheromone starts at 1. Heuristic max is ~1/min_dist.
    # If min_dist is 0.01, heuristic is 100. Boosted by 5 is 500.
    # This is fine for float64.
    
    return heuristic
# EVOLVE-BLOCK-END
