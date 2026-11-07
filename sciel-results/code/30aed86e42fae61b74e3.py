"""Heuristic matrix for ACO-TSP: MDS + MST + Convex Hull + Delaunay + 2-Opt Tour Prior."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial
import scipy.sparse.csgraph
import sys

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
    
    # Check correlation to ensure MDS is valid
    try:
        from scipy.spatial.distance import pdist, squareform
        rec_D = squareform(pdist(coords))
        corr = np.corrcoef(rec_D.flatten(), D.flatten())[0, 1]
        if corr < 0.99:
            return None
    except:
        return None
        
    return coords

def _two_opt_improve(tour, D, max_passes=50):
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
    best_len = sum(D[best_tour[i], best_tour[(i+1) % n]] for i in range(n))
    
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 1):
            for j in range(i + 1, n):
                if i == 0 and j == n - 1:
                    continue
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
        # Vectorized NN search
        dists = D[current, list(unvisited)]
        min_idx = np.argmin(dists)
        next_node = list(unvisited)[min_idx]
        visited.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    return visited

def heuristics(distance_matrix):
    """
    Return an edge-attractiveness matrix (larger = more attractive).
    """
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    
    if n < 3:
        heuristic = 1.0 / (d + 1e-9) ** 2
        return np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    # Deadline guard
    if time.time() > _DEADLINE - 2.0:
        heuristic = 1.0 / (d + 1e-9) ** 2
        return np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    # Initialize boost matrix
    # We use a multiplicative boost approach on top of a distance-based base.
    # Base heuristic: 1 / d^alpha
    # Boosts: MST, Hull, Delaunay, Best Tour
    boost_mst = np.ones((n, n))
    boost_hull = np.ones((n, n))
    boost_del = np.ones((n, n))
    boost_tour = np.ones((n, n))
    
    # 1. MDS Coordinates
    coords = _mds_coordinates(d)
    
    if coords is not None:
        try:
            # 2. MST Boost
            # scipy.sparse.csgraph.minimum_spanning_tree returns a matrix with weights
            # We just need the structure.
            # Note: MST is a tree, so n-1 edges.
            mst_graph = scipy.sparse.csgraph.minimum_spanning_tree(d)
            # mst_graph is a matrix where mst_graph[i,j] > 0 if (i,j) is in MST
            # Actually, it returns the weight. If weight > 0, edge exists.
            # But wait, MST might have 0 weight if distances are 0, which is unlikely for random points.
            # Better to check structure.
            # scipy.sparse.csgraph.minimum_spanning_tree returns a sparse matrix.
            # Let's convert to dense to check edges.
            mst_dense = mst_graph.toarray()
            # Edges in MST
            for i in range(n):
                for j in range(i+1, n):
                    if mst_dense[i, j] > 1e-9:
                        boost_mst[i, j] = 2.5
                        boost_mst[j, i] = 2.5
                        
            # 3. Convex Hull Boost
            hull = scipy.spatial.ConvexHull(coords)
            for simplex in hull.simplices:
                i, j = simplex
                boost_hull[i, j] = 3.0
                boost_hull[j, i] = 3.0
                
            # 4. Delaunay Triangulation Boost
            tri = scipy.spatial.Delaunay(coords)
            for simplex in tri.simplices:
                i, j, k = simplex
                boost_del[i, j] = 1.5
                boost_del[j, i] = 1.5
                boost_del[j, k] = 1.5
                boost_del[k, j] = 1.5
                boost_del[i, k] = 1.5
                boost_del[k, i] = 1.5
        except Exception:
            pass

    # 5. Best Tour Boost
    # Find a good tour using NN + 2-Opt
    best_tour = None
    best_len = np.inf
    
    # Try a few starting nodes
    num_starts = min(5, n)
    for start in range(num_starts):
        if time.time() > _DEADLINE - 5.0:
            break
        greedy = _greedy_tour_from(start, d)
        improved_tour, improved_len = _two_opt_improve(greedy, d, max_passes=30)
        if improved_len < best_len:
            best_len = improved_len
            best_tour = improved_tour
            
    if best_tour is None:
        # Fallback
        best_tour = _greedy_tour_from(0, d)
        best_tour, best_len = _two_opt_improve(best_tour, d, max_passes=30)
        
    # Boost edges in the best tour
    for i in range(n):
        u = best_tour[i]
        v = best_tour[(i + 1) % n]
        boost_tour[u, v] = 4.0
        boost_tour[v, u] = 4.0

    # 6. Construct Final Heuristic
    # Base: 1 / d^2
    # Combined Boost: product of boosts
    # To avoid overflow, we can normalize or just use the product.
    # The product of boosts for an edge in all structures would be 2.5 * 3.0 * 1.5 * 4.0 = 45.
    # This is a strong signal.
    
    # Ensure diagonal is 0 to avoid self-loops
    d_safe = d.copy()
    np.fill_diagonal(d_safe, 1.0) # Avoid division by zero, will be set to 0 later
    
    base_heuristic = 1.0 / (d_safe + 1e-10) ** 2
    
    # Combine boosts
    # We want edges that are in multiple structures to be very attractive.
    # Multiplicative combination is good.
    combined_boost = boost_mst * boost_hull * boost_del * boost_tour
    
    heuristic = base_heuristic * combined_boost
    
    # Set diagonal to 0
    np.fill_diagonal(heuristic, 0.0)
    
    # Handle non-finite values
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Optional: Normalize heuristic to a reasonable range to prevent numerical issues in ACO
    # The ACO probability is proportional to pheromone * heuristic.
    # If heuristic values vary by orders of magnitude, it might be okay, but let's check.
    # Max value could be large. Min is 1e-9.
    # We can normalize by the max value to keep it in [1e-9, 1] or similar?
    # Actually, the relative scale is what matters.
    # However, if the max is 100 and min is 1e-9, the ratio is 1e11. This is fine for float64.
    
    return heuristic
# EVOLVE-BLOCK-END
