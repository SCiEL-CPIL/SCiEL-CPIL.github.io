"""Heuristic matrix for ACO-TSP: Rank-based attractiveness with strong local search prior and crossing penalty."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix

_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0

def _remaining():
    return _DEADLINE - time.time()

def _mds_coordinates(D):
    """Recover 2D coordinates from a Euclidean distance matrix using Classical MDS."""
    n = D.shape[0]
    if n < 3:
        return None
    if not np.allclose(D, D.T, atol=1e-8):
        return None
    if np.any(D < -1e-9):
        return None
    J = np.eye(n) - np.ones((n, n)) / n
    D2 = D ** 2
    B = -0.5 * (J @ D2 @ J)
    B = (B + B.T) / 2.0
    try:
        w, V = np.linalg.eigh(B)
    except Exception:
        return None
    idx = np.argsort(w)[::-1]
    w = w[idx]
    V = V[:, idx]
    if w[0] < 1e-9:
        return None
    coords = V[:, :2] * np.sqrt(np.maximum(w[:2], 0))
    # Verify reconstruction quality
    rec_D = np.sqrt(np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2, axis=2))
    corr = np.corrcoef(rec_D.flatten(), D.flatten())[0, 1]
    if np.isnan(corr) or corr < 0.99:
        return None
    return coords

def _build_greedy_tour(d, start=0):
    """Build a greedy nearest-neighbor tour starting from a given node."""
    n = d.shape[0]
    visited = [start]
    unvisited = set(range(n))
    unvisited.remove(start)
    current = start
    while unvisited:
        min_dist = np.inf
        next_node = -1
        for node in unvisited:
            if d[current, node] < min_dist:
                min_dist = d[current, node]
                next_node = node
        visited.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    return visited

def _two_opt_improve(tour, D, max_iters=1000):
    """
    Apply 2-opt and Or-Opt local search to improve a tour.
    """
    n = len(tour)
    if n < 4:
        return list(tour)
    best_tour = list(tour)
    best_len = sum(D[best_tour[i], best_tour[(i + 1) % n]] for i in range(n))
    improved = True
    iter_count = 0
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        
        # 2-opt pass
        for i in range(n - 1):
            for j in range(i + 1, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = best_tour[i], best_tour[i + 1]
                c, d_ = best_tour[j], best_tour[(j + 1) % n]
                old = D[a, b] + D[c, d_]
                new = D[a, c] + D[b, d_]
                if new < old - 1e-12:
                    best_tour[i + 1:j + 1] = best_tour[i + 1:j + 1][::-1]
                    best_len += new - old
                    improved = True
                    break
            if improved:
                break
        
        # Or-Opt pass (move 1 node)
        if not improved:
            for i in range(n):
                if i == 0:
                    continue
                a = best_tour[i - 1]
                b = best_tour[i]
                c = best_tour[(i + 1) % n]
                for j in range(n):
                    if j == i or j == i - 1 or j == (i + 1) % n:
                        continue
                    x = best_tour[j - 1]
                    y = best_tour[j]
                    old = D[a, b] + D[b, c] + D[x, y]
                    new = D[a, c] + D[x, b] + D[b, y]
                    if new < old - 1e-12:
                        best_tour.pop(i)
                        best_tour.insert(j, b)
                        best_len += new - old
                        improved = True
                        break
                if improved:
                    break
        
        # Or-Opt pass (move 2 nodes)
        if not improved:
            for i in range(n - 1):
                if i == 0:
                    continue
                a = best_tour[i - 1]
                b = best_tour[i]
                c = best_tour[i + 1]
                d_ = best_tour[(i + 2) % n]
                for j in range(n):
                    if j in [i, i+1, i-1, (i+2)%n]:
                        continue
                    x = best_tour[j - 1]
                    y = best_tour[j]
                    old = D[a, b] + D[b, c] + D[c, d_] + D[x, y]
                    new = D[a, c] + D[c, b] + D[b, d_] + D[x, y]
                    if new < old - 1e-12:
                        best_tour.pop(i+1)
                        best_tour.pop(i)
                        best_tour.insert(j, b)
                        best_tour.insert(j, c)
                        best_len += new - old
                        improved = True
                        break
                if improved:
                    break

    return best_tour

def _segments_cross(p1, p2, p3, p4):
    """Check if segment p1-p2 crosses segment p3-p4 using cross products."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def _tour_crossings(tour, coords):
    """Count the number of crossing edges in a tour."""
    n = len(tour)
    if n < 4:
        return 0
    crossings = 0
    edges = []
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        edges.append((u, v))
    
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            u1, v1 = edges[i]
            u2, v2 = edges[j]
            # Skip adjacent edges (they share a vertex)
            if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                continue
            if _segments_cross(coords[u1], coords[v1], coords[u2], coords[v2]):
                crossings += 1
    return crossings

def heuristics(distance_matrix):
    """
    Return an edge-attractiveness matrix (larger = more attractive).
    
    Strategy:
    1. Rank-based heuristic: Inverse rank of distance.
    2. Strong Local Search Prior: Boost edges from a 2-opt+Or-Opt optimized tour significantly.
    3. Geometric Pruning: If MDS works, penalize edges not in Delaunay/Hull/KNN.
    """
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    
    if n < 3:
        heuristic = 1.0 / (d + 1e-9) ** 4
        return np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Deadline guard
    if _remaining() < 5.0:
        heuristic = 1.0 / (d + 1e-10) ** 3
        np.fill_diagonal(heuristic, 0.0)
        return np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # 1. Rank-based heuristic
    # Flatten, sort, assign ranks
    flat_d = d.flatten()
    # Mask diagonal
    diag_mask = np.eye(n, dtype=bool)
    flat_d_masked = flat_d.copy()
    flat_d_masked[diag_mask.flatten()] = np.inf
    
    # Ranks: 1 is best (shortest)
    ranks = np.argsort(np.argsort(flat_d_masked)) + 1
    ranks = ranks.reshape(n, n)
    
    # Heuristic: 1/rank, scaled
    heuristic = 1.0 / ranks
    heuristic[diag_mask] = 0.0
    
    # 2. Strong Local Search Prior
    best_tour = None
    best_len = np.inf
    
    if _remaining() > 10.0:
        # Try multiple starts - more starts for better tour prior
        starts = [0, 1, 2, 3, 4, n//4, n//2, 3*n//4, n-2, n-1]
        for start in starts:
            if _remaining() < 5.0:
                break
            greedy_tour = _build_greedy_tour(d, start=start)
            improved_tour = _two_opt_improve(greedy_tour, d, max_iters=500)
            tour_len = sum(d[improved_tour[i], improved_tour[(i + 1) % n]] for i in range(n))
            if tour_len < best_len:
                best_len = tour_len
                best_tour = improved_tour
        
        if best_tour is not None:
            # Massive boost for edges in the best tour
            # This effectively tells the ants: "This is a very good tour, stick to these edges"
            tour_boost = 100.0
            for i in range(n):
                u = best_tour[i]
                v = best_tour[(i + 1) % n]
                heuristic[u, v] *= tour_boost
                heuristic[v, u] *= tour_boost
    
    # 3. Geometric Pruning / Boosting
    coords = None
    if _remaining() > 5.0:
        coords = _mds_coordinates(d)
    
    if coords is not None and _remaining() > 2.0:
        try:
            # KNN boost
            k = min(6, n - 1)
            knn_boost = 2.0
            for i in range(n):
                dists = d[i].copy()
                dists[i] = np.inf
                nn_indices = np.argpartition(dists, k)[:k]
                for j in nn_indices:
                    heuristic[i, j] *= knn_boost
                    heuristic[j, i] *= knn_boost
            
            # Delaunay Triangulation boost
            tri = scipy.spatial.Delaunay(coords)
            delaunay_boost = 3.0
            for simplex in tri.simplices:
                i, j, k = simplex
                heuristic[i, j] *= delaunay_boost
                heuristic[j, i] *= delaunay_boost
                heuristic[j, k] *= delaunay_boost
                heuristic[k, j] *= delaunay_boost
                heuristic[i, k] *= delaunay_boost
                heuristic[k, i] *= delaunay_boost
            
            # Convex Hull boost
            hull = scipy.spatial.ConvexHull(coords)
            hull_boost = 10.0
            for simplex in hull.simplices:
                i, j = simplex
                heuristic[i, j] *= hull_boost
                heuristic[j, i] *= hull_boost
            
            # Crossing edge penalty: penalize edges that are likely to cross
            # Use a simpler approach: for each pair of non-adjacent nodes,
            # check if the segment would cross with many other potential edges
            # This is expensive, so we use a proxy: penalize long edges that
            # are not in the Delaunay triangulation
            if _remaining() > 1.0:
                # Build a set of Delaunay edges for quick lookup
                delaunay_edges = set()
                for simplex in tri.simplices:
                    i, j, k = simplex
                    delaunay_edges.add((min(i,j), max(i,j)))
                    delaunay_edges.add((min(j,k), max(j,k)))
                    delaunay_edges.add((min(i,k), max(i,k)))
                
                # Penalize non-Delaunay edges that are relatively long
                # This helps avoid crossing edges
                for i in range(n):
                    for j in range(i+1, n):
                        if (i, j) not in delaunay_edges:
                            # Check if this edge is long relative to local neighborhood
                            # Simple heuristic: if distance is above median, penalize slightly
                            pass
                
                # Better approach: use the best tour to identify crossing patterns
                # and penalize edges that would create crossings
                if best_tour is not None:
                    # Count how many times each edge appears in a "crossing" configuration
                    # with the best tour edges
                    tour_edges = set()
                    for i in range(n):
                        u = best_tour[i]
                        v = best_tour[(i + 1) % n]
                        tour_edges.add((min(u,v), max(u,v)))
                    
                    # For each non-tour edge, check if it crosses any tour edge
                    crossing_penalty = 0.1
                    for i in range(n):
                        for j in range(i+1, n):
                            if (i, j) in tour_edges:
                                continue
                            # Check if edge (i,j) crosses any tour edge
                            crosses = False
                            for (u, v) in tour_edges:
                                if i == u or i == v or j == u or j == v:
                                    continue
                                if _segments_cross(coords[i], coords[j], coords[u], coords[v]):
                                    crosses = True
                                    break
                            if crosses:
                                heuristic[i, j] *= crossing_penalty
                                heuristic[j, i] *= crossing_penalty
                        
        except Exception:
            pass
    
    # Ensure diagonal is 0
    np.fill_diagonal(heuristic, 0.0)
    
    # Final check: non-finite or non-positive become 1e-9
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
