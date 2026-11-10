import time
import numpy as np

_DEADLINE_FRACTION = 0.95
_WALL_LIMIT = 330.0
_start = time.time()


def _deadline_left():
    return _WALL_LIMIT * _DEADLINE_FRACTION - (time.time() - _start)


def _two_opt(d, order):
    """
    Improve a tour using the 2-opt heuristic.
    """
    n = len(order)
    if n < 4:
        return order, float(sum(d[order[i], order[(i + 1) % n]] for i in range(n)))
    
    def tour_len(o):
        return float(np.sum(d[o, np.roll(o, -1)]))

    best = list(order)
    best_len = tour_len(best)
    
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Edges to remove: (i-1, i) and (j, j+1)
                # Edges to add: (i-1, j) and (i, j+1)
                a = best[i - 1]
                b = best[i]
                c = best[j]
                e = best[(j + 1) % n]
                
                delta = (d[a, c] + d[b, e]) - (d[a, b] + d[c, e])
                
                if delta < -1e-10:
                    best[i:j + 1] = best[i:j + 1][::-1]
                    best_len += delta
                    improved = True
                    
    return best, best_len


def _three_opt(d, order):
    """
    Improve a tour using the 3-opt heuristic (reverse a segment).
    """
    n = len(order)
    if n < 5:
        return order, float(sum(d[order[i], order[(i + 1) % n]] for i in range(n)))
    
    def tour_len(o):
        return float(np.sum(d[o, np.roll(o, -1)]))

    best = list(order)
    best_len = tour_len(best)
    
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    # Try reversing segment [i..j]
                    a = best[i - 1]
                    b = best[i]
                    c = best[j]
                    e = best[(j + 1) % n]
                    
                    delta = (d[a, c] + d[b, e]) - (d[a, b] + d[c, e])
                    
                    if delta < -1e-10:
                        best[i:j + 1] = best[i:j + 1][::-1]
                        best_len += delta
                        improved = True
                        
    return best, best_len


def _or_opt(d, order):
    """
    Improve a tour using Or-opt (moving 1 node).
    """
    n = len(order)
    if n < 4:
        return order, float(sum(d[order[i], order[(i + 1) % n]] for i in range(n)))

    def tour_len(o):
        return float(np.sum(d[o, np.roll(o, -1)]))
        
    best = list(order)
    best_len = tour_len(best)
    
    improved = True
    while improved:
        improved = False
        for i in range(n):
            u = best[i]
            prev_node = best[(i - 1) % n]
            next_node = best[(i + 1) % n]
            
            len_without_u = best_len - d[prev_node, u] - d[u, next_node] + d[prev_node, next_node]
            
            temp_order = best[:]
            temp_order.pop(i)
            
            best_in_local = None
            best_len_local = np.inf
            
            for k in range(len(temp_order)):
                prev_k = temp_order[(k - 1) % len(temp_order)]
                next_k = temp_order[k % len(temp_order)]
                
                delta_insert = d[prev_k, u] + d[u, next_k] - d[prev_k, next_k]
                new_len = len_without_u + delta_insert
                
                if new_len < best_len_local:
                    best_len_local = new_len
                    best_in_local = temp_order[:]
                    best_in_local.insert(k, u)
                    
            if best_in_local is not None and best_len_local < best_len - 1e-10:
                best = best_in_local
                best_len = best_len_local
                improved = True

    return best, best_len

def _swap_opt(d, order):
    """
    Improve a tour by swapping two non-adjacent nodes.
    """
    n = len(order)
    if n < 4:
        return order, float(sum(d[order[i], order[(i + 1) % n]] for i in range(n)))

    def tour_len(o):
        return float(np.sum(d[o, np.roll(o, -1)]))
        
    best = list(order)
    best_len = tour_len(best)
    
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i + 2, n):
                if j == i + 1:
                    continue
                if i == 0 and j == n - 1:
                    continue
                
                u = best[i]
                v = best[j]
                
                prev_u = best[(i - 1) % n]
                next_u = best[(i + 1) % n]
                prev_v = best[(j - 1) % n]
                next_v = best[(j + 1) % n]
                
                delta = (d[prev_v, u] + d[u, next_v] + d[prev_u, v] + d[v, next_u]) - \
                        (d[prev_u, u] + d[u, next_u] + d[prev_v, v] + d[v, next_v])
                
                if delta < -1e-10:
                    best[i], best[j] = best[j], best[i]
                    best_len += delta
                    improved = True
                    
    return best, best_len

def _nearest_neighbor_tour(d, start=0):
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


def _find_best_tour(d):
    n = d.shape[0]
    if n <= 2:
        return list(range(n))
        
    best_order = None
    best_len = np.inf
    
    starts = list(range(n))
    
    for s in starts:
        if _deadline_left() < 10.0:
            break
            
        tour = _nearest_neighbor_tour(d, start=s)
        tour, l = _two_opt(d, tour)
        tour, l = _three_opt(d, tour)
        tour, l = _or_opt(d, tour)
        tour, l = _swap_opt(d, tour)
        tour, l = _two_opt(d, tour)
        
        if l < best_len:
            best_len = l
            best_order = tour
            
    if best_order is None:
        best_order = _nearest_neighbor_tour(d, start=0)
        
    return best_order


def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for the ACO solver.
    Uses a smooth inverse-distance base with a moderate tour boost to guide
    ACO while preserving exploration of geometrically plausible alternatives.
    """
    d = np.asarray(distance_matrix, dtype=float)
    n = d.shape[0]
    
    if n <= 1:
        return d.copy()
    
    # Base heuristic: 1 / distance^3 (moderate suppression, preserves gradient)
    d_safe = np.where(d > 1e-10, d, 1e10)
    H = 1.0 / (d_safe ** 3)
    
    # Ensure diagonal is small
    np.fill_diagonal(H, 1e-9)
    
    # Boost edges from a good tour with a moderate multiplier
    if n > 2 and _deadline_left() > 15.0:
        tour = _find_best_tour(d)
        
        # Use a boost that's 5x the max base heuristic to allow exploration
        max_base = np.max(H)
        boost_val = max_base * 5.0
        
        for i in range(n):
            u = tour[i]
            v = tour[(i + 1) % n]
            H[u, v] = boost_val
            H[v, u] = boost_val
            
    # Sanitize per contract
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    return H
