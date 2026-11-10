"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4 (proper intersection only)."""
    def orient(a, b, c):
        val = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(val) < 1e-10:
            return 0
        return 1 if val > 0 else -1
    
    o1 = orient(p1, p2, p3)
    o2 = orient(p1, p2, p4)
    o3 = orient(p3, p4, p1)
    o4 = orient(p3, p4, p2)
    
    if o1 != o2 and o3 != o4:
        return True
    return False


def heuristics(distance_matrix):
    """
    Compute an edge-attractiveness matrix for ACO TSP using a geometric
    non-crossing heuristic. 
    
    Structural approach: Instead of local distance-based components, we compute
    a global geometric penalty based on how many other edges each candidate edge
    would cross. In 2D, optimal TSP tours are non-crossing, so edges that cross
    many other potential edges are strongly suppressed. Combined with a smooth
    distance-based attractance, this gives a heuristic that naturally guides
    ants toward non-crossing, short tours.
    """
    n = distance_matrix.shape[0]
    deadline = time.time() + 270  # Safety margin
    
    # Recover 2D coordinates from distance matrix via classical MDS
    # D^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i . x_j
    D2 = distance_matrix ** 2
    # Center: double centering
    ones = np.ones((n, n))
    row_sum = D2.sum(axis=1, keepdims=True)
    col_sum = D2.sum(axis=0, keepdims=True)
    total_sum = D2.sum() / n
    B = -0.5 * (D2 - row_sum / n - col_sum / n + total_sum / n)
    
    # Eigenvalue decomposition for 2D embedding
    eigvals, eigvecs = np.linalg.eigh(B)
    # Take top 2 eigenvalues (largest)
    idx = np.argsort(eigvals)[::-1][:2]
    eigvals_top = eigvals[idx]
    eigvecs_top = eigvecs[:, idx]
    
    # Ensure positive eigenvalues
    eigvals_top = np.maximum(eigvals_top, 0)
    coords = eigvecs_top * np.sqrt(eigvals_top)
    
    # Verify: reconstruct distances and check quality
    # If MDS is poor, fall back to a simpler heuristic
    recon_dist = np.sqrt((coords[:, 0]**2 + coords[:, 1]**2)[:, None] + 
                          (coords[:, 0]**2 + coords[:, 1]**2)[None, :] - 
                          2 * coords @ coords.T)
    recon_dist = np.abs(recon_dist)
    # Diagonal should be 0
    np.fill_diagonal(recon_dist, 0)
    
    # Check MDS quality
    off_diag = distance_matrix[~np.eye(n, dtype=bool)]
    recon_off = recon_dist[~np.eye(n, dtype=bool)]
    if len(off_diag) > 0:
        rel_err = np.mean(np.abs(recon_off - off_diag)) / (np.mean(off_diag) + 1e-12)
    else:
        rel_err = 1.0
    
    # Base heuristic: inverse distance to power 2.5 (strong suppression of long edges)
    base = 1.0 / (distance_matrix + 1e-9) ** 2.5
    
    if rel_err < 0.5:
        # MDS quality is good enough - use geometric crossing penalty
        # Count crossings for each edge
        crossing_count = np.zeros((n, n), dtype=np.float64)
        
        for i in range(n):
            for j in range(i + 1, n):
                p1 = coords[i]
                p2 = coords[j]
                cross_cnt = 0.0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    for l in range(k + 1, n):
                        if l == i or l == j:
                            continue
                        p3 = coords[k]
                        p4 = coords[l]
                        if _segments_intersect(p1, p2, p3, p4):
                            # Weight by inverse of the crossing edge length
                            cross_dist = distance_matrix[k, l]
                            cross_cnt += 1.0 / (cross_dist + 1e-9)
                
                crossing_count[i, j] = cross_cnt
                crossing_count[j, i] = cross_cnt
        
        # Normalize crossing counts
        max_cross = crossing_count.max()
        if max_cross > 1e-12:
            crossing_count_norm = crossing_count / max_cross
        else:
            crossing_count_norm = crossing_count
        
        # Penalty: edges with more crossings are less attractive
        # Use exponential decay for smooth penalty
        crossing_penalty = np.exp(-1.5 * crossing_count_norm)
        
        heuristic = base * crossing_penalty
    else:
        # MDS failed - fall back to pure distance heuristic with k-NN boost
        k = 5
        knn_boost = np.zeros_like(distance_matrix)
        for i in range(n):
            dists = distance_matrix[i].copy()
            dists[i] = np.inf
            nn_indices = np.argsort(dists)[:k]
            for rank, j in enumerate(nn_indices):
                weight = 1.0 - (rank / k)
                knn_boost[i, j] = weight
                knn_boost[j, i] = max(knn_boost[j, i], weight)
        
        heuristic = base * (1.0 + 0.5 * knn_boost)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
