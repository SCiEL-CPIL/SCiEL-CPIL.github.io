"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix):
    """
    Heuristic based on CMDS-recovered coordinates, 2-Opt reference tour,
    and explicit geometric crossing counts to enforce planarity.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    n = distance_matrix.shape[0]
    
    # 1. Recover 2D coordinates using Classical MDS
    # This allows us to perform geometric crossing checks
    def get_coords(dm):
        n = dm.shape[0]
        D2 = dm ** 2
        row_means = np.mean(D2, axis=1, keepdims=True)
        col_means = np.mean(D2, axis=0, keepdims=True)
        grand_mean = np.mean(D2)
        B = -0.5 * (D2 - row_means - col_means.T + grand_mean)
        
        # Eigendecomposition
        # Use eigh for symmetric matrix
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        
        # Sort by descending eigenvalue
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Take top 2 dimensions
        # Handle potential negative eigenvalues (numerical noise)
        ev = np.maximum(eigenvalues[:2], 0)
        coords = eigenvectors[:, :2] * np.sqrt(ev)
        
        return coords

    coords = get_coords(distance_matrix)
    
    # 2. Generate Reference Tour: Nearest Neighbor + 2-Opt
    # NN Tour
    visited = [False] * n
    tour = [0]
    visited[0] = True
    current = 0
    
    for _ in range(n - 1):
        min_dist = np.inf
        next_node = -1
        for j in range(n):
            if not visited[j]:
                d = distance_matrix[current, j]
                if d < min_dist:
                    min_dist = d
                    next_node = j
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
        
    # 2-Opt Improvement
    def tour_length(t):
        total = 0.0
        for i in range(n):
            total += distance_matrix[t[i], t[(i + 1) % n]]
        return total

    current_len = tour_length(tour)
    improved = True
    max_iter = 1000
    iter_count = 0
    
    while improved and iter_count < max_iter:
        if time.time() > deadline:
            break
        improved = False
        iter_count += 1
        for i in range(n - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[(j + 1) % n]
                
                old_len = distance_matrix[a, b] + distance_matrix[c, d]
                new_len = distance_matrix[a, c] + distance_matrix[b, d]
                
                if new_len < old_len - 1e-10:
                    tour[i:j+1] = tour[i:j+1][::-1]
                    current_len -= old_len - new_len
                    improved = True
        if time.time() > deadline:
            break

    # 3. Compute Crossing Counts
    # For each edge (u, v), count how many edges in the reference tour it crosses.
    # This explicitly encodes planarity: edges that don't cross the good tour are preferred.
    
    def orientation(p1, p2, p3):
        val = (p2[1]-p1[1])*(p3[0]-p2[0]) - (p2[0]-p1[0])*(p3[1]-p2[1])
        if val > 0: return 1
        if val < 0: return 2
        return 0

    def segments_cross(p1, p2, p3, p4):
        o1 = orientation(p1, p2, p3)
        o2 = orientation(p1, p2, p4)
        o3 = orientation(p3, p4, p1)
        o4 = orientation(p3, p4, p2)
        
        if o1 != o2 and o3 != o4:
            return True
        return False

    cross_counts = np.zeros((n, n), dtype=int)
    
    # Get edges of the reference tour
    tour_edges = []
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        if u < v:
            tour_edges.append((u, v))
        else:
            tour_edges.append((v, u))
            
    # For each pair (u, v), check crossing with all tour edges
    # Optimization: Only check u < v
    for u in range(n):
        if time.time() > deadline:
            break
        for v in range(u + 1, n):
            count = 0
            p1 = coords[u]
            p2 = coords[v]
            for (a, b) in tour_edges:
                p3 = coords[a]
                p4 = coords[b]
                if segments_cross(p1, p2, p3, p4):
                    count += 1
            cross_counts[u, v] = count
            cross_counts[v, u] = count
            
    # 4. Build Heuristic Matrix
    # Base: 1 / distance^5
    safe_dist = np.where(distance_matrix == 0, 1e-9, distance_matrix)
    base_heuristic = 1.0 / (safe_dist ** 5)
    
    # Planarity Factor: Inversely proportional to crossing count
    # If an edge crosses 0 edges of the reference tour, factor = 1.0
    # If it crosses 1, factor = 1/6 ~ 0.16
    # If it crosses 2, factor = 1/11 ~ 0.09
    # This strongly penalizes edges that are geometrically incompatible with the reference tour
    lambda_cross = 5.0
    planarity_factor = 1.0 / (1.0 + lambda_cross * cross_counts)
    
    heuristic = base_heuristic * planarity_factor
    
    # Boost edges in the reference tour
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        heuristic[u, v] *= 50.0
        heuristic[v, u] *= 50.0
        
    # Boost k-NN edges (k=3)
    k = 3
    for i in range(n):
        dists = distance_matrix[i].copy()
        dists[i] = np.inf
        nn_indices = np.argsort(dists)[:k]
        for j in nn_indices:
            heuristic[i, j] *= 5.0
            heuristic[j, i] *= 5.0 # Symmetric boost
            
    # Ensure no zeros or negatives
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
