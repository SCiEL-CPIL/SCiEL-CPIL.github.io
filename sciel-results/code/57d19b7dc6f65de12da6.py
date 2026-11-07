"""Heuristic matrix for ant colony optimization of the traveling salesman problem."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay, KDTree
from scipy.linalg import eigh


def _mds_coordinates(d, n):
    """Compute 2D MDS coordinates from a distance matrix."""
    try:
        # Double centering
        ones = np.ones((n, n)) / n
        J = np.eye(n) - ones
        B = -0.5 * J @ (d ** 2) @ J
        # Eigen decomposition
        eigvals, eigvecs = eigh(B)
        # Take the two largest eigenvalues
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        # Ensure non-negative eigenvalues
        eigvals = np.maximum(eigvals, 0)
        coords = eigvecs[:, :2] * np.sqrt(eigvals[:2])
        return coords
    except Exception:
        return None


def _two_opt_improve(tour, dist, max_passes=50):
    """Improve a tour using 2-opt moves."""
    tour = list(tour)
    n = len(tour)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if j == n - 1 and i == 1:
                    continue
                # Calculate the change in tour length
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[(j + 1) % n]
                old = dist[a, b] + dist[c, d]
                new = dist[a, c] + dist[b, d]
                if new + 1e-10 < old:
                    # Reverse the segment
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
    return tour


def _greedy_tour(dist, start=0, n=None):
    """Construct a tour using nearest neighbor heuristic."""
    if n is None:
        n = dist.shape[0]
    visited = [False] * n
    tour = [start]
    visited[start] = True
    current = start
    for _ in range(n - 1):
        # Find nearest unvisited node
        min_dist = np.inf
        next_node = -1
        for j in range(n):
            if not visited[j] and dist[current, j] < min_dist:
                min_dist = dist[current, j]
                next_node = j
        if next_node == -1:
            break
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    return tour


def _detect_crossing_edges(coords, tour):
    """Detect crossing edges in a tour using MDS coordinates."""
    n = len(tour)
    crossing_mask = np.zeros((n, n), dtype=bool)
    
    def segments_intersect(p1, p2, p3, p4):
        """Check if segment p1-p2 intersects with segment p3-p4."""
        def orient(a, b, c):
            val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
            if val > 1e-10:
                return 1
            elif val < -1e-10:
                return -1
            return 0
        
        o1 = orient(p1, p2, p3)
        o2 = orient(p1, p2, p4)
        o3 = orient(p3, p4, p1)
        o4 = orient(p3, p4, p2)
        
        if o1 != o2 and o3 != o4:
            return True
        return False
    
    # Check all pairs of non-adjacent edges
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            edge1 = (tour[i], tour[(i + 1) % n])
            edge2 = (tour[j], tour[(j + 1) % n])
            
            # Skip if edges share a node
            if edge1[0] == edge2[0] or edge1[0] == edge2[1] or edge1[1] == edge2[0] or edge1[1] == edge2[1]:
                continue
            
            p1 = coords[edge1[0]]
            p2 = coords[edge1[1]]
            p3 = coords[edge2[0]]
            p4 = coords[edge2[1]]
            
            if segments_intersect(p1, p2, p3, p4):
                crossing_mask[edge1[0], edge1[1]] = True
                crossing_mask[edge1[1], edge1[0]] = True
                crossing_mask[edge2[0], edge2[1]] = True
                crossing_mask[edge2[1], edge2[0]] = True
    
    return crossing_mask


def _fiedler_penalty(d, n):
    """Compute Fiedler vector penalty for long-range connections."""
    try:
        # Create graph with inverse-distance weights
        d_safe = d.copy()
        np.fill_diagonal(d_safe, np.inf)
        weights = np.where(d_safe > 0, 1.0 / d_safe, 0.0)
        
        # Compute graph Laplacian
        degree = np.sum(weights, axis=1)
        L = np.diag(degree) - weights
        
        # Compute Fiedler vector (second smallest eigenvalue)
        eigvals, eigvecs = eigh(L)
        fiedler = eigvecs[:, 1]
        
        # Compute penalty based on difference in Fiedler coordinates
        diff = np.abs(fiedler[:, None] - fiedler[None, :])
        # Normalize and create penalty (larger diff = larger penalty)
        max_diff = np.max(diff)
        if max_diff > 0:
            penalty = 1.0 / (1.0 + 5.0 * diff / max_diff)
        else:
            penalty = np.ones((n, n))
        
        return penalty
    except Exception:
        return np.ones((n, n))


def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 280  # 300s limit, keep 20s margin
    
    d = np.array(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    
    # Ensure the matrix is square and valid
    if d.shape[0] != d.shape[1]:
        return np.ones_like(d)
    
    # Work on a copy to avoid modifying the original
    d_safe = d.copy()
    np.fill_diagonal(d_safe, np.inf)
    
    # Compute MDS coordinates for geometric reasoning
    coords = _mds_coordinates(d_safe, n)
    
    # Build heuristic from multiple components
    # 1. Base distance-based heuristic
    K = 15
    alpha = 2.5
    eta_base = np.full((n, n), 1e-9)
    
    for i in range(n):
        indices = np.argsort(d_safe[i])[:K]
        for j in indices:
            if d_safe[i, j] > 0 and np.isfinite(d_safe[i, j]):
                eta_base[i, j] = 1.0 / (d_safe[i, j] ** alpha)
            else:
                eta_base[i, j] = 1e-9
    
    eta_base = np.maximum(eta_base, eta_base.T)
    
    # 2. Delaunay triangulation edges
    try:
        if coords is not None:
            tri = Delaunay(coords)
            delaunay_mask = np.zeros((n, n), dtype=bool)
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        u, v = simplex[i], simplex[j]
                        delaunay_mask[u, v] = True
                        delaunay_mask[v, u] = True
            
            # Boost Delaunay edges
            eta_base[delaunay_mask] *= 1.5
    except Exception:
        pass
    
    # 3. Multi-start greedy tours with 2-opt
    best_tour = None
    best_len = np.inf
    num_starts = 10
    
    for start in range(min(num_starts, n)):
        if time.time() > deadline - 30:
            break
        tour = _greedy_tour(d_safe, start=start, n=n)
        tour = _two_opt_improve(tour, d_safe, max_passes=30)
        
        # Calculate tour length
        length = 0
        for i in range(n):
            length += d_safe[tour[i], tour[(i + 1) % n]]
        
        if length < best_len:
            best_len = length
            best_tour = tour
    
    # 4. Boost edges in the best tour
    if best_tour is not None:
        tour_mask = np.zeros((n, n), dtype=bool)
        for i in range(n):
            u, v = best_tour[i], best_tour[(i + 1) % n]
            tour_mask[u, v] = True
            tour_mask[v, u] = True
        eta_base[tour_mask] *= 2.0
    
    # 5. Crossing edge penalty
    if coords is not None and best_tour is not None:
        if time.time() > deadline - 20:
            crossing_mask = np.zeros((n, n), dtype=bool)
        else:
            crossing_mask = _detect_crossing_edges(coords, best_tour)
        eta_base[crossing_mask] *= 0.3
    
    # 6. Fiedler vector penalty for long-range connections
    if time.time() > deadline - 10:
        fiedler_pen = np.ones((n, n))
    else:
        fiedler_pen = _fiedler_penalty(d_safe, n)
    eta_base *= fiedler_pen
    
    # Final safety check
    eta_base = np.where(np.isfinite(eta_base) & (eta_base > 0), eta_base, 1e-9)
    np.fill_diagonal(eta_base, 1e-9)
    
    return eta_base
# EVOLVE-BLOCK-END
