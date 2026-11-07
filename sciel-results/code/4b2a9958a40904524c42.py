"""Heuristic matrix for ACO-TSP: Direct structural approach with MDS + geometry + tour prior."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0

def _mds_coordinates(D):
    """
    Recover 2D coordinates from a Euclidean distance matrix using Classical MDS.
    Returns coordinates or None if it fails.
    """
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
    except:
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

def _build_greedy_tour(d):
    """
    Build a greedy nearest-neighbor tour starting from node 0.
    """
    n = d.shape[0]
    visited = [0]
    unvisited = set(range(1, n))
    current = 0
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

def _two_opt_improve(tour, D, max_iters=300):
    """
    Apply 2-opt local search to improve a tour.
    Returns (best_tour, removed_edges_set) where removed_edges are edges that were
    present in the input tour but removed by 2-opt moves.
    """
    n = len(tour)
    if n < 4:
        return list(tour), set()
    
    best_tour = list(tour)
    
    # Track edges that get removed
    removed_edges = set()
    
    # Compute initial length
    best_len = sum(D[best_tour[i], best_tour[(i+1) % n]] for i in range(n))
    
    improved = True
    iter_count = 0
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        for i in range(n - 1):
            for j in range(i + 1, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = best_tour[i], best_tour[i+1]
                c, d_ = best_tour[j], best_tour[(j+1) % n]
                
                old = D[a, b] + D[c, d_]
                new = D[a, c] + D[b, d_]
                
                if new < old - 1e-12:
                    # Record removed edges
                    removed_edges.add((min(a,b), max(a,b)))
                    removed_edges.add((min(c,d_), max(c,d_)))
                    best_tour[i+1:j+1] = best_tour[i+1:j+1][::-1]
                    best_len += new - old
                    improved = True
                    break
            if improved:
                break
    
    return best_tour, removed_edges

def _do_tours_cross(tour1, tour2, D, coords=None):
    """
    Check if any edge in tour1 crosses any edge in tour2.
    Uses geometric crossing test if coords available, otherwise angle-based heuristic.
    Returns a penalty factor (lower = more crossings = worse).
    """
    n = len(tour1)
    if n < 4:
        return 1.0
    
    if coords is None:
        return 1.0
    
    # Get edges from tour1
    edges1 = []
    for i in range(n):
        u = tour1[i]
        v = tour1[(i+1) % n]
        edges1.append((u, v))
    
    # Check each edge in tour2 against edges in tour1
    crossings = 0
    for i in range(n):
        a = tour2[i]
        b = tour2[(i+1) % n]
        pa = coords[a]
        pb = coords[b]
        
        for (c, d_) in edges1:
            if c == a or c == b or d_ == a or d_ == b:
                continue  # Skip shared endpoints
            pc = coords[c]
            pd = coords[d_]
            
            # Check if segments (pa,pb) and (pc,pd) cross
            # Using cross product test
            def cross(o, a, b):
                return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
            
            d1 = cross(pc, pd, pa)
            d2 = cross(pc, pd, pb)
            d3 = cross(pa, pb, pc)
            d4 = cross(pa, pb, pd)
            
            if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
               ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                crossings += 1
    
    # Penalty: exponential decay with number of crossings
    if crossings == 0:
        return 1.0
    else:
        return 1.0 / (1.0 + 2.0 * crossings)

def heuristics(distance_matrix):
    """
    Return an edge-attractiveness matrix (larger = more attractive).
    
    Direct approach:
    1. Base: inverse distance to a power.
    2. Build multiple greedy tours from different starts, 2-opt each, keep best.
    3. Boost edges from the best tour.
    4. Penalize edges that cross with the best tour (geometric non-crossing prior).
    5. Penalize edges that were removed during 2-opt (bad edge signal).
    6. Boost k-nearest-neighbor edges.
    7. If MDS recovers coordinates, boost Delaunay triangulation edges and convex hull edges.
    """
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    
    if n < 3:
        heuristic = 1.0 / (d + 1e-9) ** 4
        return np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    # Deadline guard
    if time.time() > _DEADLINE - 3.0:
        heuristic = 1.0 / (d + 1e-10) ** 3
        np.fill_diagonal(heuristic, 0.0)
        return np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    # 1. Base heuristic: inverse cube of distance
    heuristic = 1.0 / (d + 1e-10) ** 3

    # 2. Build multiple greedy tours from different starts, 2-opt each, keep best
    best_tour = None
    best_len = np.inf
    all_removed_edges = set()
    
    # Try multiple starting nodes for greedy tours
    start_nodes = [0, 1, n//2, n-1]
    for start in start_nodes:
        if time.time() > _DEADLINE - 5.0:
            break
        
        # Build greedy tour from this start
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
        
        # 2-opt improve
        improved_tour, removed = _two_opt_improve(visited, d, max_iters=150)
        all_removed_edges.update(removed)
        
        tour_len = sum(d[improved_tour[i], improved_tour[(i+1) % n]] for i in range(n))
        if tour_len < best_len:
            best_len = tour_len
            best_tour = improved_tour
    
    if best_tour is None:
        best_tour = _build_greedy_tour(d)
        best_tour, _ = _two_opt_improve(best_tour, d, max_iters=150)

    # 3. Boost edges from the best tour
    tour_boost = 10.0
    for i in range(n):
        u = best_tour[i]
        v = best_tour[(i + 1) % n]
        heuristic[u, v] *= tour_boost
        heuristic[v, u] *= tour_boost

    # 4. Penalize edges that cross with the best tour
    coords = _mds_coordinates(d)
    if coords is not None and time.time() < _DEADLINE - 3.0:
        # Build a penalty matrix for crossing edges
        crossing_penalty = np.ones((n, n))
        tour_edges = [(best_tour[i], best_tour[(i+1) % n]) for i in range(n)]
        
        for i in range(n):
            for j in range(i+1, n):
                if i == j:
                    continue
                # Check if edge (i,j) crosses any tour edge
                pa = coords[i]
                pb = coords[j]
                for (c, d_) in tour_edges:
                    if c == i or c == j or d_ == i or d_ == j:
                        continue
                    pc = coords[c]
                    pd = coords[d_]
                    
                    def cross(o, a, b):
                        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
                    
                    d1 = cross(pc, pd, pa)
                    d2 = cross(pc, pd, pb)
                    d3 = cross(pa, pb, pc)
                    d4 = cross(pa, pb, pd)
                    
                    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
                       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                        crossing_penalty[i, j] *= 0.3
                        crossing_penalty[j, i] *= 0.3
                        break  # One crossing is enough to penalize
        
        heuristic *= crossing_penalty

    # 5. Penalize edges that were removed during 2-opt (bad edge signal)
    if len(all_removed_edges) > 0:
        removed_penalty = 0.4
        for (u, v) in all_removed_edges:
            heuristic[u, v] *= removed_penalty
            heuristic[v, u] *= removed_penalty

    # 6. K-nearest neighbors boost
    k = 6
    knn_boost = 2.5
    for i in range(n):
        dists = d[i].copy()
        dists[i] = np.inf
        nn_indices = np.argpartition(dists, k)[:k]
        for j in nn_indices:
            heuristic[i, j] *= knn_boost
            heuristic[j, i] *= knn_boost

    # 7. Geometric boosts if MDS succeeded
    if coords is not None and time.time() < _DEADLINE - 2.0:
        try:
            # Delaunay Triangulation boost
            tri = scipy.spatial.Delaunay(coords)
            delaunay_boost = 2.0
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
            hull_boost = 5.0
            for simplex in hull.simplices:
                i, j = simplex
                heuristic[i, j] *= hull_boost
                heuristic[j, i] *= hull_boost
                
        except Exception:
            pass

    # Ensure diagonal is 0
    np.fill_diagonal(heuristic, 0.0)
    
    # Final check: non-finite or non-positive become 1e-9
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
