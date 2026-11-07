"""Heuristic matrix for ACO-TSP: MDS + Delaunay + Convex Hull + Multi-start 2-opt tour prior + crossing penalty."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial
import sys

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 295.0  # 5s safety margin

def _mds_coordinates(D):
    """
    Recover 2D coordinates from a Euclidean distance matrix using Classical MDS.
    Returns coordinates or None if it fails.
    """
    n = D.shape[0]
    if n < 3:
        return None
    
    # Check for symmetry and non-negativity
    if not np.allclose(D, D.T):
        return None
    if np.any(D < -1e-9):
        return None

    # Classical MDS
    # B = -0.5 * J * D^2 * J
    J = np.eye(n) - np.ones((n, n)) / n
    D2 = D ** 2
    B = -0.5 * (J @ D2 @ J)
    
    # Symmetrize B to remove numerical noise
    B = (B + B.T) / 2.0
    
    try:
        w, V = np.linalg.eigh(B)
    except:
        return None
        
    # Sort eigenvalues in descending order
    idx = np.argsort(w)[::-1]
    w = w[idx]
    V = V[:, idx]
    
    # The first two eigenvalues should be positive for 2D points
    if w[0] < 1e-9:
        return None
        
    # Coordinates
    coords = V[:, :2] * np.sqrt(np.maximum(w[:2], 0))
    
    # Relax tolerance check to avoid false negatives from numerical noise
    # If points are truly Euclidean, error should be tiny.
    # We'll just check if the reconstruction is "reasonable" (correlation > 0.99)
    from scipy.spatial.distance import pdist, squareform
    rec_D = squareform(pdist(coords))
    
    # Check correlation
    if np.corrcoef(rec_D.flatten(), D.flatten())[0, 1] < 0.99:
        return None
        
    return coords

def _two_opt_improve(tour, D):
    """
    Apply 2-opt local search to improve a tour.
    tour: list of node indices forming a closed tour
    D: distance matrix
    Returns improved tour and its length.
    """
    n = len(tour)
    if n < 4:
        return tour, D[tour[0], tour[1]] + D[tour[1], tour[2]] + D[tour[2], tour[0]]
    
    best_tour = list(tour)
    best_len = 0.0
    # Compute initial length
    for i in range(n):
        best_len += D[best_tour[i], best_tour[(i+1) % n]]
    
    improved = True
    max_iters = 300
    iter_count = 0
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        for i in range(n - 1):
            for j in range(i + 1, n):
                if i == 0 and j == n - 1:
                    continue
                # Try reversing segment i+1..j
                a, b = best_tour[i], best_tour[i+1]
                c, d_ = best_tour[j], best_tour[(j+1) % n]
                
                old = D[a, b] + D[c, d_]
                new = D[a, c] + D[b, d_]
                
                if new < old - 1e-12:
                    best_tour[i+1:j+1] = best_tour[i+1:j+1][::-1]
                    best_len += new - old
                    improved = True
                    break
            if improved:
                break
    
    return best_tour, best_len


def _greedy_tour_from(start, D):
    """Build a nearest-neighbor tour starting from a given node."""
    n = D.shape[0]
    visited = [start]
    unvisited = set(range(n))
    unvisited.remove(start)
    current = start
    while unvisited:
        min_dist = np.inf
        next_node = -1
        for node in unvisited:
            if D[current, node] < min_dist:
                min_dist = D[current, node]
                next_node = node
        visited.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    return visited


def _detect_crossing_edges(tour, coords):
    """
    Detect edges in a tour that cross other edges (in 2D).
    Returns a set of (min(a,b), max(a,b)) pairs for crossing edges.
    """
    n = len(tour)
    if n < 4 or coords is None:
        return set()
    
    def segments_intersect(p1, p2, p3, p4):
        """Check if line segment p1-p2 intersects segment p3-p4 (proper crossing)."""
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False
    
    crossing_edges = set()
    edges = []
    for i in range(n):
        u = tour[i]
        v = tour[(i+1) % n]
        edges.append((u, v))
    
    for i in range(len(edges)):
        for j in range(i+1, len(edges)):
            # Skip adjacent edges (share a vertex)
            if j == i + 1 or (i == 0 and j == len(edges)-1):
                continue
            u1, v1 = edges[i]
            u2, v2 = edges[j]
            # Skip if they share a vertex
            if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                continue
            
            p1 = coords[u1]
            p2 = coords[v1]
            p3 = coords[u2]
            p4 = coords[v2]
            
            if segments_intersect(p1, p2, p3, p4):
                crossing_edges.add((min(u1, v1), max(u1, v1)))
                crossing_edges.add((min(u2, v2), max(u2, v2)))
    
    return crossing_edges

def heuristics(distance_matrix):
    """
    Return an edge-attractiveness matrix (larger = more attractive).
    
    Strategy:
    1. Base heuristic: Inverse power of distance.
    2. K-Nearest Neighbors Boost.
    3. Recover coordinates via MDS for Delaunay/Convex Hull boosts.
    4. Inject 2-opt NN tour edges as a strong structural prior.
    """
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    
    if n < 3:
        # Trivial case
        heuristic = 1.0 / (d + 1e-9) ** 4
        return np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    # Deadline guard
    if time.time() > _DEADLINE - 2.0:
        heuristic = 1.0 / (d + 1e-9) ** 4
        return np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    # 1. Base Heuristic: Inverse 3rd power
    heuristic = 1.0 / (d + 1e-10) ** 3
    
    # 2. K-Nearest Neighbors Boost
    k = 5
    knn_boost = 2.0
    for i in range(n):
        dists = d[i].copy()
        dists[i] = np.inf
        nn_indices = np.argpartition(dists, k)[:k]
        for j in nn_indices:
            heuristic[i, j] *= knn_boost
            heuristic[j, i] *= knn_boost

    # 3. Try to recover coordinates and use geometry
    coords = _mds_coordinates(d)
    
    if coords is not None:
        try:
            # Delaunay Triangulation
            tri = scipy.spatial.Delaunay(coords)
            
            delaunay_boost = 1.5
            for simplex in tri.simplices:
                i, j, k = simplex
                heuristic[i, j] *= delaunay_boost
                heuristic[j, i] *= delaunay_boost
                heuristic[j, k] *= delaunay_boost
                heuristic[k, j] *= delaunay_boost
                heuristic[i, k] *= delaunay_boost
                heuristic[k, i] *= delaunay_boost
                
            # Convex Hull
            hull = scipy.spatial.ConvexHull(coords)
            
            hull_boost = 3.0
            for simplex in hull.simplices:
                i, j = simplex
                heuristic[i, j] *= hull_boost
                heuristic[j, i] *= hull_boost
                
        except Exception:
            pass

    # 4. Multi-start greedy + 2-opt for best tour prior
    # Try multiple starting nodes to find a better tour
    best_tour = None
    best_len = np.inf
    num_starts = min(10, n)  # Try up to 10 different starting nodes
    
    for start in range(num_starts):
        greedy = _greedy_tour_from(start, d)
        improved_tour, improved_len = _two_opt_improve(greedy, d)
        if improved_len < best_len:
            best_len = improved_len
            best_tour = improved_tour
    
    if best_tour is None:
        # Fallback: just use NN from node 0
        best_tour = _greedy_tour_from(0, d)
        best_tour, best_len = _two_opt_improve(best_tour, d)
    
    # Boost edges in the best 2-opt improved tour
    tour_boost = 5.0
    for i in range(n):
        u = best_tour[i]
        v = best_tour[(i + 1) % n]
        heuristic[u, v] *= tour_boost
        heuristic[v, u] *= tour_boost
    
    # 5. Penalize crossing edges detected via MDS coordinates
    if coords is not None:
        crossing_edges = _detect_crossing_edges(best_tour, coords)
        if crossing_edges:
            crossing_penalty = 0.3  # Reduce attractiveness of crossing edges
            for (a, b) in crossing_edges:
                heuristic[a, b] *= crossing_penalty
                heuristic[b, a] *= crossing_penalty
    
    # Ensure diagonal is 0 (or very small)
    np.fill_diagonal(heuristic, 0.0)
    
    # Final check for finiteness and positivity
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
