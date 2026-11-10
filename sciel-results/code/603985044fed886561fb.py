import numpy as np

def heuristics(distance_matrix):
    n = distance_matrix.shape[0]
    # Base heuristic: inverse squared distance
    # Add small epsilon to avoid division by zero
    base = 1.0 / (distance_matrix + 1e-9) ** 2
    
    # 1. Construct NN tour
    visited = np.zeros(n, dtype=bool)
    tour = [0]
    visited[0] = True
    for _ in range(n - 1):
        current = tour[-1]
        # Find nearest unvisited
        # Mask visited nodes with infinity
        dists = distance_matrix[current].copy()
        dists[visited] = np.inf
        next_node = np.argmin(dists)
        tour.append(next_node)
        visited[next_node] = True
    
    # 2. Improve with 2-opt
    # Calculate tour length
    def tour_length(t):
        total = 0.0
        for i in range(n):
            total += distance_matrix[t[i], t[(i + 1) % n]]
        return total
    
    current_len = tour_length(tour)
    
    # Simple 2-opt improvement
    # We do a few passes until no improvement or max passes
    max_passes = 10
    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                # If i and j are adjacent, skip (no change)
                if j == i + 1:
                    continue
                # Edges involved: (i, i+1) and (j, j+1)
                # Note: indices in tour list
                a = tour[i]
                b = tour[i + 1]
                c = tour[j]
                d = tour[(j + 1) % n]
                
                # Current connection: a-b and c-d
                # New connection: a-c and b-d
                # Wait, 2-opt reverses segment i+1 to j.
                # Original: ... a, b ... c, d ...
                # After reverse: ... a, c ... b, d ...
                # So we replace edges (a,b) and (c,d) with (a,c) and (b,d).
                
                d_ab = distance_matrix[a, b]
                d_cd = distance_matrix[c, d]
                d_ac = distance_matrix[a, c]
                d_bd = distance_matrix[b, d]
                
                if d_ac + d_bd < d_ab + d_cd - 1e-9:
                    # Perform reversal
                    tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                    current_len = current_len - d_ab - d_cd + d_ac + d_bd
                    improved = True
        if not improved:
            break
            
    # 3. Extract edges from improved tour
    # H starts as base
    H = base.copy()
    
    # Boost tour edges
    boost_tour = 5.0
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        H[u, v] *= boost_tour
        H[v, u] *= boost_tour
        
    # 4. k-NN boost
    k = 3
    boost_knn = 2.0
    for i in range(n):
        # Get distances from i to all others
        dists_i = distance_matrix[i].copy()
        dists_i[i] = np.inf # Exclude self
        # Get k smallest
        # np.argpartition is faster for k smallest
        k_indices = np.argpartition(dists_i, k)[:k]
        for j in k_indices:
            H[i, j] *= boost_knn
            H[j, i] *= boost_knn
            
    # Ensure finite and positive
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    return H
