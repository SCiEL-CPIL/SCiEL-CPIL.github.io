"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _orient(a, b, c):
    """Orientation of triplet (a, b, c): 1 for CCW, -1 for CW, 0 for collinear."""
    val = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if abs(val) < 1e-10:
        return 0
    return 1 if val > 0 else -1


def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4 (proper intersection only)."""
    o1 = _orient(p1, p2, p3)
    o2 = _orient(p1, p2, p4)
    o3 = _orient(p3, p4, p1)
    o4 = _orient(p3, p4, p2)
    if o1 != o2 and o3 != o4:
        return True
    return False


def _nearest_neighbor_tour(dist):
    """Construct a nearest-neighbor tour starting from node 0."""
    n = dist.shape[0]
    visited = np.zeros(n, dtype=bool)
    tour = [0]
    visited[0] = True
    current = 0
    for _ in range(n - 1):
        # Find nearest unvisited neighbor
        unvisited = ~visited
        unvisited[current] = True  # Exclude current
        nearest = np.argmin(np.where(unvisited, dist[current], np.inf))
        tour.append(nearest)
        visited[nearest] = True
        current = nearest
    return tour


def _two_opt_tour(tour, dist, max_passes=10):
    """Improve tour using 2-opt local search."""
    n = len(tour)
    if n < 4:
        return tour
    
    best_tour = list(tour)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                # Calculate delta for reversing tour[i+1..j]
                a, b = best_tour[i], best_tour[i + 1]
                c, d = best_tour[j], best_tour[(j + 1) % n]
                old_len = dist[a][b] + dist[c][d]
                new_len = dist[a][c] + dist[b][d]
                if new_len < old_len - 1e-12:
                    best_tour[i + 1:j + 1] = best_tour[i + 1:j + 1][::-1]
                    improved = True
    return best_tour


def _three_opt_tour(tour, dist, max_passes=5):
    """Improve tour using 3-opt local search."""
    n = len(tour)
    if n < 5:
        return tour
    
    best_tour = list(tour)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    if i == 0 and k == n - 1:
                        continue
                    a, b = best_tour[i], best_tour[i + 1]
                    c, d = best_tour[j], best_tour[j + 1]
                    e, f = best_tour[k], best_tour[(k + 1) % n]
                    
                    # Option 1: reverse segment [i+1..j]
                    old_len = dist[a][b] + dist[c][d]
                    new_len = dist[a][c] + dist[b][d]
                    if new_len < old_len - 1e-12:
                        best_tour[i + 1:j + 1] = best_tour[i + 1:j + 1][::-1]
                        improved = True
                        continue
                    
                    # Option 2: reverse segment [j+1..k]
                    old_len = dist[c][d] + dist[e][f]
                    new_len = dist[c][f] + dist[d][e]
                    if new_len < old_len - 1e-12:
                        best_tour[j + 1:k + 1] = best_tour[j + 1:k + 1][::-1]
                        improved = True
                        continue
    return best_tour


def heuristics(distance_matrix):
    """
    Compute an edge-attractiveness matrix for ACO TSP using a dynamic
    repulsive heuristic derived from a 3-opt refined nearest-neighbor tour.
    
    Strategy:
    1. Build a high-quality reference tour using NN + 2-opt + 3-opt
    2. Identify edges in the reference tour (high attractiveness)
    3. Identify edges that cross reference tour edges (low attractiveness)
    4. Combine with inverse distance base
    
    This encodes planarity constraints from a locally optimal tour structure.
    """
    n = distance_matrix.shape[0]
    start_time = time.time()
    deadline = start_time + 270  # Safety margin
    
    # Step 1: Build reference tour
    nn_tour = _nearest_neighbor_tour(distance_matrix)
    two_opt_tour = _two_opt_tour(nn_tour, distance_matrix, max_passes=10)
    ref_tour = _three_opt_tour(two_opt_tour, distance_matrix, max_passes=5)
    
    # Step 2: Extract edges from reference tour
    ref_edges = set()
    for idx in range(n):
        u = ref_tour[idx]
        v = ref_tour[(idx + 1) % n]
        ref_edges.add((min(u, v), max(u, v)))
    
    # Step 3: Base heuristic - inverse distance to power 3.0
    base = 1.0 / (distance_matrix + 1e-9) ** 3.0
    
    # Step 4: Compute crossing penalty based on reference tour edges
    # For each candidate edge, count how many reference tour edges it crosses
    crossing_penalty = np.ones((n, n), dtype=np.float64)
    
    # Try to recover 2D coordinates for geometric crossing detection
    D2 = distance_matrix ** 2
    row_sum = D2.sum(axis=1, keepdims=True)
    col_sum = D2.sum(axis=0, keepdims=True)
    total_sum = D2.sum() / n
    B = -0.5 * (D2 - row_sum / n - col_sum / n + total_sum / n)
    
    eigvals, eigvecs = np.linalg.eigh(B)
    idx_top = np.argsort(eigvals)[::-1][:2]
    eigvals_top = np.maximum(eigvals[idx_top], 0)
    coords = eigvecs[:, idx_top] * np.sqrt(eigvals_top)
    
    # Verify MDS quality
    recon_dist_sq = (coords[:, 0]**2 + coords[:, 1]**2)[:, None] + \
                    (coords[:, 0]**2 + coords[:, 1]**2)[None, :] - \
                    2 * coords @ coords.T
    recon_dist = np.sqrt(np.abs(recon_dist_sq))
    np.fill_diagonal(recon_dist, 0)
    
    off_diag_mask = ~np.eye(n, dtype=bool)
    rel_err = np.mean(np.abs(recon_dist[off_diag_mask] - distance_matrix[off_diag_mask])) / \
              (np.mean(distance_matrix[off_diag_mask]) + 1e-12)
    
    if rel_err < 0.5 and time.time() < deadline:
        # Good MDS - use geometric crossing detection with reference tour
        ref_edge_list = list(ref_edges)
        
        for i in range(n):
            if time.time() > deadline - 30:
                break
            for j in range(i + 1, n):
                p1, p2 = coords[i], coords[j]
                cross_count = 0.0
                for (u, v) in ref_edge_list:
                    if u == i or u == j or v == i or v == j:
                        continue
                    p3, p4 = coords[u], coords[v]
                    if _segments_intersect(p1, p2, p3, p4):
                        cross_count += 1.0
                
                # Exponential penalty: more crossings = less attractive
                crossing_penalty[i, j] = np.exp(-2.0 * cross_count)
                crossing_penalty[j, i] = crossing_penalty[i, j]
    
    # Step 5: Boost reference tour edges
    ref_boost = np.ones((n, n), dtype=np.float64)
    for (u, v) in ref_edges:
        ref_boost[u, v] = 15.0
        ref_boost[v, u] = 15.0
    
    # Combine all components
    heuristic = base * crossing_penalty * ref_boost
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
