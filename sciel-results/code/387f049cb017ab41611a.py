"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute edge-attractiveness matrix using graph-theoretic centrality features.
    
    Uses a weighted combination of:
    - Prize-weighted eigenvector centrality
    - Betweenness centrality (approximate)
    - Degree centrality
    - Distance-decay proximity to origin
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = len(prizes)
    
    # Feature 1: Prize-weighted eigenvector centrality
    # Build adjacency based on inverse distance, weighted by prizes
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_dist = 1.0 / (distance_matrix + 1e-9)
    inv_dist[np.diag_indices(n)] = 0.0
    
    # Normalize rows for eigenvector computation
    row_sums = inv_dist.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    norm_adj = inv_dist / row_sums
    
    # Prize-weighted adjacency
    prize_weighted_adj = norm_adj * prizes[None, :]
    
    # Compute eigenvector centrality via power iteration
    eigvec = np.ones(n) / np.sqrt(n)
    for _ in range(50):
        new_eigvec = prize_weighted_adj @ eigvec
        new_norm = np.linalg.norm(new_eigvec)
        if new_norm < 1e-15:
            break
        new_eigvec /= new_norm
        if np.linalg.norm(new_eigvec - eigvec) < 1e-10:
            break
        eigvec = new_eigvec
    
    # Feature 2: Approximate betweenness centrality
    # Use shortest path counts via BFS-like approach on thresholded graph
    betweenness = np.zeros(n)
    threshold = np.percentile(distance_matrix[~np.eye(n, dtype=bool)], 50)
    adj_bool = distance_matrix < threshold
    adj_bool[np.diag_indices(n)] = False
    
    for s in range(n):
        # BFS from s
        dist = np.full(n, np.inf)
        count = np.zeros(n)
        dist[s] = 0
        count[s] = 1
        queue = [s]
        visited = set([s])
        
        while queue:
            next_queue = []
            for u in queue:
                for v in range(n):
                    if adj_bool[u, v]:
                        if dist[v] == np.inf:
                            dist[v] = dist[u] + 1
                            count[v] = count[u]
                            next_queue.append(v)
                        elif dist[v] == dist[u] + 1:
                            count[v] += count[u]
            queue = next_queue
        
        # Accumulate betweenness
        for v in range(n):
            if v != s and count[v] > 0:
                betweenness[v] += 1.0 / count[v]
    
    # Normalize betweenness
    if betweenness.max() > 0:
        betweenness /= betweenness.max()
    
    # Feature 3: Degree centrality (weighted by inverse distance)
    degree = inv_dist.sum(axis=1)
    if degree.max() > 0:
        degree /= degree.max()
    
    # Feature 4: Proximity to origin (node 0)
    origin_proximity = 1.0 / (distance_matrix[:, 0] + 1e-9)
    if origin_proximity.max() > 0:
        origin_proximity /= origin_proximity.max()
    
    # Combine features with learned weights
    # Weights chosen to balance structural importance
    w_eig = 0.35
    w_between = 0.25
    w_degree = 0.15
    w_origin = 0.25
    
    node_attractiveness = (
        w_eig * eigvec +
        w_between * betweenness +
        w_degree * degree +
        w_origin * origin_proximity
    )
    
    # Ensure non-negative
    node_attractiveness = np.maximum(node_attractiveness, 0)
    
    # Build edge matrix: attractiveness of destination node, decayed by distance
    # Edge (i,j) gets attractiveness of node j, weighted by how close j is to i
    with np.errstate(divide='ignore', invalid='ignore'):
        edge_proximity = 1.0 / (distance_matrix + 1e-9)
    edge_proximity[np.diag_indices(n)] = 0.0
    
    # Normalize edge proximity per row
    row_sums = edge_proximity.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    edge_proximity_norm = edge_proximity / row_sums
    
    # Final heuristic: node attractiveness * distance decay
    heuristic = node_attractiveness[None, :] * edge_proximity_norm
    
    # Ensure all entries are positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
