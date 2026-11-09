"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    n = distance_matrix.shape[0]
    
    # Helper to calculate tour length
    def get_tour_length(t):
        idx = np.array(t)
        next_idx = np.roll(idx, -1)
        return np.sum(distance_matrix[idx, next_idx])

    # 2-Opt improvement
    def two_opt(t):
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    a = t[i]
                    b = t[i + 1]
                    c = t[j]
                    d = t[(j + 1) % n]
                    
                    delta = (distance_matrix[a, c] + distance_matrix[b, d]) - (distance_matrix[a, b] + distance_matrix[c, d])
                    if delta < 0:
                        t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                        improved = True
        return t

    # Nearest Neighbor construction
    def nearest_neighbor(start_node):
        visited = np.zeros(n, dtype=bool)
        visited[start_node] = True
        tour = [start_node]
        current = start_node
        
        for _ in range(n - 1):
            dists = distance_matrix[current, :].copy()
            dists[visited] = np.inf
            next_node = np.argmin(dists)
            tour.append(next_node)
            visited[next_node] = True
            current = next_node
        return tour

    # Random insertion heuristic
    def random_insertion():
        tour = [0]
        unvisited = set(range(1, n))
        
        while unvisited:
            # Pick a random unvisited node
            node = list(unvisited)[np.random.randint(len(unvisited))]
            unvisited.remove(node)
            
            # Find best insertion position
            best_pos = 0
            best_cost = float('inf')
            for i in range(len(tour)):
                prev_node = tour[i - 1] if i > 0 else tour[-1]
                next_node = tour[i]
                cost = distance_matrix[prev_node, node] + distance_matrix[node, next_node] - distance_matrix[prev_node, next_node]
                if cost < best_cost:
                    best_cost = cost
                    best_pos = i
            tour.insert(best_pos, node)
        
        return tour

    # Find best tour using multiple methods
    best_tour = None
    best_len = float('inf')
    
    # Method 1: Multiple NN restarts
    num_restarts = 30
    for k in range(num_restarts):
        if time.time() > deadline:
            break
        start_node = np.random.randint(n)
        tour = nearest_neighbor(start_node)
        tour = two_opt(tour)
        length = get_tour_length(tour)
        
        if length < best_len:
            best_len = length
            best_tour = tour
    
    # Method 2: Random insertion with 2-opt
    for k in range(10):
        if time.time() > deadline:
            break
        tour = random_insertion()
        tour = two_opt(tour)
        length = get_tour_length(tour)
        
        if length < best_len:
            best_len = length
            best_tour = tour
    
    # If no tour was found, fallback
    if best_tour is None:
        best_tour = nearest_neighbor(0)
        best_tour = two_opt(best_tour)
        best_len = get_tour_length(best_tour)

    # Build MST for connectivity guidance
    # Use a simple Prim's algorithm starting from node 0
    def compute_mst():
        in_mst = np.zeros(n, dtype=bool)
        in_mst[0] = True
        mst_edges = []
        
        for _ in range(n - 1):
            min_dist = float('inf')
            min_u = -1
            min_v = -1
            
            for u in range(n):
                if not in_mst[u]:
                    for v in range(n):
                        if in_mst[v]:
                            if distance_matrix[u, v] < min_dist:
                                min_dist = distance_matrix[u, v]
                                min_u = u
                                min_v = v
            
            if min_u == -1:
                break
            
            in_mst[min_u] = True
            mst_edges.append((min_u, min_v))
        
        return mst_edges
    
    mst_edges = compute_mst()
    mst_edge_set = set()
    for u, v in mst_edges:
        mst_edge_set.add((u, v))
        mst_edge_set.add((v, u))

    # Build Heuristic Matrix
    diag_mask = np.eye(n, dtype=bool)
    safe_dist = distance_matrix.copy()
    safe_dist[diag_mask] = 1.0
    
    # Component 1: Inverse distance (local optimality)
    inv_dist = 1.0 / safe_dist
    
    # Component 2: MST edges (connectivity)
    mst_heuristic = np.ones_like(inv_dist)
    mst_heuristic[diag_mask] = 0
    for u, v in mst_edge_set:
        mst_heuristic[u, v] = 5.0
        mst_heuristic[v, u] = 5.0
    
    # Component 3: Best tour edges (global structure)
    tour_heuristic = np.ones_like(inv_dist)
    tour_heuristic[diag_mask] = 0
    for i in range(n):
        j = (i + 1) % n
        tour_heuristic[best_tour[i], best_tour[j]] = 10.0
        tour_heuristic[best_tour[j], best_tour[i]] = 10.0
    
    # Combine heuristics with weighted sum
    # Weight towards a blend that balances local, connectivity, and global
    base = 0.4 * inv_dist + 0.3 * mst_heuristic + 0.3 * tour_heuristic
    
    # Ensure diagonal is not attractive
    base[diag_mask] = 1e-9
    
    # Ensure all values are positive and finite
    base = np.maximum(base, 1e-9)
    base = np.where(np.isfinite(base), base, 1e-9)
    
    return base
# EVOLVE-BLOCK-END
