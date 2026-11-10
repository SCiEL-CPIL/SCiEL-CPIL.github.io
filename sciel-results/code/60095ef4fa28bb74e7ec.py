import numpy as np
import time

def heuristics(distance_matrix):
    """
    Heuristic matrix for ant colony optimization of TSP.
    
    Combines a strong inverse distance base with structural priors:
    1. Base: d^-2 (strong inverse to suppress long edges)
    2. Heuristic Tour Boost: Strong boost to edges in a NN+2Opt tour
    3. Mutual k-NN Boost: Moderate boost to mutually close neighbors
    
    The heuristic tour provides a complete valid cycle example, guiding ants
    towards forming a single large loop. The k-NN ensures local consistency.
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 330s limit
    
    try:
        dm = np.asarray(distance_matrix, dtype=float)
        n = dm.shape[0]
        
        # Check for non-finite values
        if not np.all(np.isfinite(dm)):
            raise ValueError("Distance matrix contains non-finite values")
        
        # 1. Base heuristic: inverse squared distance
        d_safe = dm + 1e-9
        base = 1.0 / (d_safe ** 2)
        
        # 2. Compute a fast heuristic tour (Nearest Neighbor + 2-Opt)
        # This provides a strong prior for edges that form a valid, short cycle.
        
        # Nearest Neighbor heuristic starting from node 0
        visited = np.zeros(n, dtype=bool)
        tour = [0]
        visited[0] = True
        current = 0
        
        for _ in range(n - 1):
            # Find nearest unvisited node
            dists = dm[current].copy()
            dists[visited] = np.inf
            next_node = np.argmin(dists)
            tour.append(next_node)
            visited[next_node] = True
            current = next_node
        
        # 2-Opt improvement
        def tour_length(t):
            total = 0.0
            for i in range(n):
                total += dm[t[i], t[(i + 1) % n]]
            return total
        
        current_len = tour_length(tour)
        
        # Simple 2-opt improvement
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
                    
                    d_ab = dm[a, b]
                    d_cd = dm[c, d]
                    d_ac = dm[a, c]
                    d_bd = dm[b, d]
                    
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
            dists_i = dm[i].copy()
            dists_i[i] = np.inf # Exclude self
            # Get k smallest
            # np.argpartition is faster for k smallest
            k_indices = np.argpartition(dists_i, k)[:k]
            for j in k_indices:
                H[i, j] *= boost_knn
                H[j, i] *= boost_knn
        
        # Ensure finite and positive
        H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
        
        # Check time
        if time.time() > deadline:
            # Fallback to simple inverse distance
            return 1.0 / (dm + 1e-9) ** 2
        
        # Ensure valid output
        if not np.all(np.isfinite(H)):
            H = np.where(np.isfinite(H), H, 1e-9)
        if not np.all(H > 0):
            H = np.where(H > 0, H, 1e-9)
        
        return H
        
    except Exception:
        # Fallback to simple inverse distance
        try:
            return 1.0 / (distance_matrix + 1e-9) ** 2
        except Exception:
            return np.ones_like(distance_matrix)
