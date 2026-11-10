import time
import numpy as np

# Explicit deadline guard
_DEADLINE_FRACTION = 0.9
_WALL_LIMIT = 330.0
_start = time.time()


def _deadline_left():
    return _WALL_LIMIT * _DEADLINE_FRACTION - (time.time() - _start)


def _nearest_neighbor_tour(d, start=0):
    """
    Construct a tour using the Nearest Neighbor heuristic starting from a specific node.
    """
    n = d.shape[0]
    visited = np.zeros(n, dtype=bool)
    order = [start]
    visited[start] = True
    cur = start
    for _ in range(n - 1):
        dist = d[cur].copy()
        dist[visited] = np.inf
        nxt = int(np.argmin(dist))
        order.append(nxt)
        visited[nxt] = True
        cur = nxt
    return order


def _two_opt(d, order, max_iters=100):
    """
    Improve a tour using the 2-opt heuristic.
    Returns the improved order and its length.
    """
    n = len(order)
    if n < 4:
        return order, float(sum(d[order[i], order[(i + 1) % n]] for i in range(n)))
    
    # Build a distance lookup via the matrix using node indices
    # order is list of node indices; closed tour order -> order -> order[0]
    def tour_len(o):
        return float(sum(d[o[i], o[(i + 1) % n]] for i in range(n)))

    best = list(order)
    best_len = tour_len(best)
    
    # Optimization: Precompute delta improvements? No, simple loop is fine for n=50.
    # Standard 2-opt: reverse segment between i and j.
    # Edges removed: (best[i-1], best[i]) and (best[j], best[(j+1)%n])
    # Edges added: (best[i-1], best[j]) and (best[i], best[(j+1)%n])
    
    for _ in range(max_iters):
        improved = False
        # i is the start of the segment to reverse (index in order)
        # We iterate i from 1 to n-2 (0-based indices in the list)
        # and j from i+1 to n-1.
        # Note: In a cyclic tour, reversing [i..j] changes edges (i-1, i) and (j, j+1) to (i-1, j) and (i, j+1).
        # Indices are modulo n.
        
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # We allow j = i+1 (swap adjacent) as it is a valid 2-opt move.
                
                a = best[i - 1]
                b = best[i]
                c = best[j]
                e = best[(j + 1) % n]
                
                # Current edges: (a, b) and (c, e)
                # New edges: (a, c) and (b, e)
                delta = (d[a, c] + d[b, e]) - (d[a, b] + d[c, e])
                
                if delta < -1e-12:
                    best[i:j + 1] = best[i:j + 1][::-1]
                    best_len += delta
                    improved = True
        
        if not improved:
            break
            
    return best, best_len


def _good_directed_tour(d):
    """
    Find a high-quality directed tour by trying multiple nearest-neighbor starts
    and applying 2-opt to each, keeping the best result.
    """
    n = d.shape[0]
    if n <= 2:
        return list(range(n)), float(sum(d[i, (i+1)%n] for i in range(n)))

    best_order = None
    best_len = np.inf
    
    # Try a few different starting nodes to escape local minima of NN
    # For n=50, trying 30 starts is cheap and effective.
    start_pool = list(range(0, min(n, 30)))
    
    # Shuffle starts to diversify? Or just sequential.
    # Since the problem is random, sequential is fine.
    
    for s in start_pool:
        if _deadline_left() < 5.0:
            break
            
        tmp = _nearest_neighbor_tour(d, start=s)
        # Rotate so s is first? Not strictly necessary for 2-opt, but consistent.
        # NN already starts at s.
        
        opt, ln = _two_opt(d, tmp, max_iters=100)
        
        if ln < best_len:
            best_len = ln
            best_order = opt
            
    if best_order is None:
        # Fallback
        best_order = _nearest_neighbor_tour(d, start=0)
        best_order, best_len = _two_opt(d, best_order, max_iters=10)
        
    return best_order, best_len


def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for the ACO solver.
    Uses a combination of inverse distance and a strong directed boost for edges
    present in a near-optimal tour found via multi-start NN + 2-opt.
    """
    d = np.asarray(distance_matrix, dtype=float)
    n = d.shape[0]
    
    # Base heuristic: inverse distance
    # Add epsilon to avoid division by zero.
    # Diagonal is 0 in distance matrix usually, so 1/(0+eps) is huge.
    # We will handle diagonal separately.
    base = 1.0 / (d + 1e-9)
    
    # Ensure diagonal is not attractive (ants shouldn't pick themselves)
    # The task says "Heuristic entries that are not positive become 1e-9".
    # If we set diagonal to 0, it becomes 1e-9.
    # But usually, we want diagonal to be low.
    # Let's set diagonal to 0 for now, then sanitize at the end.
    np.fill_diagonal(base, 0.0)
    
    # Find a reference directed tour
    bias = np.zeros_like(d)
    if n > 2 and _deadline_left() > 10.0:
        order, _ = _good_directed_tour(d)
        for i in range(n):
            u = order[i]
            v = order[(i + 1) % n]
            # Use symmetric bias to allow the tour to be traversed in either direction.
            # This is often more robust for ACO on symmetric TSPs.
            bias[u, v] = 1.0
            bias[v, u] = 1.0
            
    # Blend: strong boost on the good tour edges.
    # Symmetric bias means H is symmetric.
    # Boost = 100.0 to ensure dominance.
    
    boost = 100.0
    H = base + boost * bias
    
    # Sanitize per contract:
    # 1. Non-finite values -> 1e-9
    # 2. Non-positive values -> 1e-9
    # 3. Diagonal should be low. base diagonal was 0. bias diagonal is 0 (no self-loops in tour).
    # So H diagonal is 0. It will become 1e-9.
    
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    return H
