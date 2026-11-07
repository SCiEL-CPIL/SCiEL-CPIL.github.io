"""Heuristic matrix for ACO-TSP: MST + multi-start 2-opt + crossing penalty + geometry."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial

_START_TIME = time.time()
_DEADLINE = _START_TIME + 285.0

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

def _two_opt_improve(tour, D, max_iters=2000):
    """Apply 2-opt local search to improve a tour. Uses or-opt for 3-edge moves."""
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
        # or-opt pass (move a single node)
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
    return best_tour

def _segments_cross(p1, p2, p3, p4):
    """Check if segment p1-p2 crosses segment p3-p4."""
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
    """
    Return a set of edges (as frozensets) that cross in the tour.
    """
    n = len(tour)
    crossing_edges = set()
    for i in range(n):
        a, b = tour[i], tour[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            c, d = tour[j], tour[(j + 1) % n]
            if _segments_cross(coords[a], coords[b], coords[c], coords[d]):
                crossing_edges.add(frozenset((a, b)))
                crossing_edges.add(frozenset((c, d)))
    return crossing_edges

def heuristics(distance_matrix):
    """
    Return an edge-attractiveness matrix (larger = more attractive).
    
    Strategy:
    1. Base: inverse distance to a power.
    2. MST edges get high boost for connectivity.
    3. Multi-start greedy + 2-opt gives a strong tour prior.
    4. Crossing edges get penalized.
    5. KNN and geometric (Delaunay, hull) boosts if MDS works.
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
    
    # 1. Base heuristic: inverse cube of distance
    heuristic = 1.0 / (d + 1e-10) ** 3
    
    # 2. MST boost
    if _remaining() > 3.0:
        try:
            from scipy.sparse.csgraph import minimum_spanning_tree
            from scipy.sparse import csr_matrix
            D_for_mst = d.copy()
            np.fill_diagonal(D_for_mst, np.inf)
            D_sparse = csr_matrix(D_for_mst)
            mst = minimum_spanning_tree(D_sparse)
            mst_dense = mst.toarray()
            mst_edges = np.argwhere(mst_dense > 0)
            mst_boost = 8.0
            for i, j in mst_edges:
                heuristic[i, j] *= mst_boost
                heuristic[j, i] *= mst_boost
        except Exception:
            pass
    
    # 3. Multi-start greedy + 2-opt for tour prior
    best_tour = None
    best_len = np.inf
    if _remaining() > 3.0:
        # Try multiple starting points
        starts = list(range(0, n, max(1, n // 5)))
        if 0 not in starts:
            starts.insert(0, 0)
        for start in starts:
            if _remaining() < 2.0:
                break
            greedy_tour = _build_greedy_tour(d, start=start)
            improved_tour = _two_opt_improve(greedy_tour, d, max_iters=400)
            tour_len = sum(d[improved_tour[i], improved_tour[(i + 1) % n]] for i in range(n))
            if tour_len < best_len:
                best_len = tour_len
                best_tour = improved_tour
        
        if best_tour is not None:
            tour_boost = 15.0
            for i in range(n):
                u = best_tour[i]
                v = best_tour[(i + 1) % n]
                heuristic[u, v] *= tour_boost
                heuristic[v, u] *= tour_boost
    
    # 4. KNN boost
    if _remaining() > 2.0:
        k = min(6, n - 1)
        knn_boost = 3.0
        for i in range(n):
            dists = d[i].copy()
            dists[i] = np.inf
            nn_indices = np.argpartition(dists, k)[:k]
            for j in nn_indices:
                heuristic[i, j] *= knn_boost
                heuristic[j, i] *= knn_boost
    
    # 5. Try MDS for geometric boosts
    coords = None
    if _remaining() > 3.0:
        coords = _mds_coordinates(d)
    
    if coords is not None and _remaining() > 2.0:
        try:
            # Delaunay Triangulation boost
            tri = scipy.spatial.Delaunay(coords)
            delaunay_boost = 5.0
            for simplex in tri.simplices:
                i, j, k = simplex
                heuristic[i, j] *= delaunay_boost
                heuristic[j, i] *= delaunay_boost
                heuristic[j, k] *= delaunay_boost
                heuristic[k, j] *= delaunay_boost
                heuristic[i, k] *= delaunay_boost
                heuristic[k, i] *= delaunay_boost
            
            # Convex Hull boost – hull edges are always in the optimal TSP tour
            hull = scipy.spatial.ConvexHull(coords)
            hull_boost = 30.0
            for simplex in hull.simplices:
                i, j = simplex
                heuristic[i, j] *= hull_boost
                heuristic[j, i] *= hull_boost
        except Exception:
            pass
    
    # 6. Crossing penalty: penalize edges that cross in the best tour
    if coords is not None and best_tour is not None and _remaining() > 1.0:
        try:
            crossing_edges = _tour_crossings(best_tour, coords)
            crossing_penalty = 0.05
            for edge in crossing_edges:
                a, b = tuple(edge)
                heuristic[a, b] *= crossing_penalty
                heuristic[b, a] *= crossing_penalty
        except Exception:
            pass
    
    # Ensure diagonal is 0
    np.fill_diagonal(heuristic, 0.0)
    
    # Final check: non-finite or non-positive become 1e-9
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
