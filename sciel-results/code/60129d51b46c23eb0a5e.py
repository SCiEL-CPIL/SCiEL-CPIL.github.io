import time
import math
import random

# Pre-define neighbors for fast edge counting
# Triangular lattice neighbors for (a, b) are:
# (a+1, b), (a-1, b), (a, b+1), (a, b-1), (a+1, b-1), (a-1, b+1)
_NEIGHBORS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

def _hexdist(a, b):
    """Calculate hexagonal distance from origin (0,0)"""
    return (abs(a) + abs(b) + abs(a + b)) // 2

def _count_edges(points):
    """Count the number of pairs at distance exactly 1"""
    point_set = set(points)
    count = 0
    for p in points:
        a, b = p
        for da, db in _NEIGHBORS:
            if (a + da, b + db) in point_set:
                count += 1
    return count // 2  # Each edge is counted twice

def _generate_hexagon(n):
    """
    Generate a list of n points in a hexagonal shape centered at (0,0).
    This creates a very compact shape which is likely close to optimal.
    """
    candidates = []
    max_r = 10 
    for a in range(-max_r, max_r + 1):
        for b in range(-max_r, max_r + 1):
            if _hexdist(a, b) <= max_r:
                x = a + 0.5 * b
                y = math.sqrt(3) / 2 * b
                angle = math.atan2(y, x)
                candidates.append((_hexdist(a, b), angle, a, b))
    
    candidates.sort(key=lambda x: (x[0], x[1]))
    
    result = [(a, b) for _, _, a, b in candidates[:n]]
    return result

def _local_search(points, n, deadline):
    """
    Perform systematic local search to maximize edges.
    Tries all possible moves: remove point i, add point c.
    """
    current_points = list(points)
    current_set = set(current_points)
    current_edges = _count_edges(current_points)
    
    point_neighbors = {}
    for p in current_points:
        a, b = p
        nbrs = []
        for da, db in _NEIGHBORS:
            np = (a + da, b + db)
            if np in current_set:
                nbrs.append(np)
        point_neighbors[p] = nbrs

    pool = set()
    for a in range(-12, 13):
        for b in range(-12, 13):
            if _hexdist(a, b) <= 12:
                pool.add((a, b))
    pool = list(pool)
    
    best_points = current_points[:]
    best_edges = current_edges
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                return best_points
            p_i = current_points[i]
            
            edges_lost = len(point_neighbors[p_i])
            
            for p_c in pool:
                if p_c == p_i or p_c in current_set:
                    continue
                
                a_c, b_c = p_c
                edges_gained = 0
                for da, db in _NEIGHBORS:
                    np = (a_c + da, b_c + db)
                    if np in current_set and np != p_i:
                        edges_gained += 1
                
                if edges_gained > edges_lost:
                    new_edges = current_edges - edges_lost + edges_gained
                    
                    for nbr in point_neighbors[p_i]:
                        if p_i in point_neighbors[nbr]:
                            point_neighbors[nbr].remove(p_i)
                    del point_neighbors[p_i]
                    current_set.discard(p_i)
                    
                    current_points[i] = p_c
                    current_set.add(p_c)
                    
                    point_neighbors[p_c] = []
                    a_c, b_c = p_c
                    for da, db in _NEIGHBORS:
                        np = (a_c + da, b_c + db)
                        if np in current_set:
                            point_neighbors[p_c].append(np)
                            if p_c not in point_neighbors[np]:
                                point_neighbors[np].append(p_c)
                    
                    current_edges = new_edges
                    improved = True
                    
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_points = current_points[:]
                    
                    if current_edges >= 76:
                        return current_points
                    
                    break
            
            if improved:
                break

    return best_points

def construct():
    deadline = time.time() + 100  # Safety margin
    n = 32
    
    # 1. Generate a high-quality initial guess
    initial_points = _generate_hexagon(n)
    
    # Check if already optimal
    edges = _count_edges(initial_points)
    if edges >= 76:
        return [[a, b] for a, b in initial_points]
    
    # 2. Refine with local search
    refined_points = _local_search(initial_points, n, deadline)
    
    final_edges = _count_edges(refined_points)
    
    # 3. If not optimal, try a few perturbed restarts
    if final_edges < 76 and time.time() < deadline - 10:
        rng = random.Random(42)
        best_points = refined_points
        best_edges = final_edges
        
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            
            pts = list(best_points)
            pset = set(pts)
            
            min_deg = float('inf')
            min_idx = -1
            for i, p in enumerate(pts):
                deg = 0
                a, b = p
                for da, db in _NEIGHBORS:
                    if (a+da, b+db) in pset:
                        deg += 1
                if deg < min_deg:
                    min_deg = deg
                    min_idx = i
            
            pool = [(a,b) for a in range(-15, 16) for b in range(-15, 16) if _hexdist(a,b) <= 15]
            candidates = [p for p in pool if p not in pset]
            if candidates:
                new_p = rng.choice(candidates)
                pts[min_idx] = new_p
                
                refined = _local_search(pts, n, deadline - 5)
                e = _count_edges(refined)
                
                if e > best_edges:
                    best_edges = e
                    best_points = refined
                    if e >= 76:
                        break

    return [[a, b] for a, b in best_points]
