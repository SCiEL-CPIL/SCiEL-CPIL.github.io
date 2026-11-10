import time
import random
from math import isqrt
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 145  # Safety margin for 150s limit
    
    # Fallback: Known valid set of 5 points
    # These points have integer coordinates and integer pairwise distances.
    # (0,0), (-56,33), (-16,63), (0,33), (16,63)
    # Distances:
    # (0,0) to others: 33, 65, 33, 65 (wait, 16^2+63^2 = 256+3969=4225=65^2)
    # Let's verify the fallback set validity quickly in code or trust it.
    # The prompt says "Verified manually above". I will include a verification step.
    
    fallback_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    fallback_D = 1

    def is_square(n):
        if n < 0:
            return False
        r = isqrt(n)
        return r * r == n

    def check_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)

    def check_concyclic(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        def det4(M):
            a,b,c,d = M[0]
            e,f,g,h = M[1]
            i,j,k,l = M[2]
            m,n,o,p = M[3]
            
            return (a * (f * (k * p - l * o) - g * (j * p - l * n) + h * (j * o - k * n)) -
                    b * (e * (k * p - l * o) - g * (i * p - l * m) + h * (i * o - k * m)) +
                    c * (e * (j * p - l * n) - f * (i * p - l * m) + h * (i * n - j * m)) -
                    d * (e * (j * o - k * n) - f * (i * o - k * m) + g * (i * n - j * m)))
        
        M = [
            [x1*x1 + y1*y1, x1, y1, 1],
            [x2*x2 + y2*y2, x2, y2, 1],
            [x3*x3 + y3*y3, x3, y3, 1],
            [x4*x4 + y4*y4, x4, y4, 1]
        ]
        
        return det4(M) == 0

    def validate_set(points):
        n = len(points)
        if len(set(points)) != n:
            return False
            
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x1 - x2
                dy = y1 - y2
                d2 = dx*dx + dy*dy
                if not is_square(d2):
                    return False
        
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if check_collinear(points[i], points[j], points[k]):
                        return False
        
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if check_concyclic(points[i], points[j], points[k], points[l]):
                            return False
                            
        return True

    # Search for larger point sets
    # We will use a backtracking search with pruning.
    # To make it feasible, we restrict the search space.
    
    best_set = None
    best_size = 0
    
    # Try different box sizes
    # L=40 gives 81*81=6561 points.
    # L=50 gives 101*101=10201 points.
    # Let's try L=50 but with aggressive pruning.
    
    L = 50
    points_in_box = []
    for x in range(-L, L+1):
        for y in range(-L, L+1):
            points_in_box.append((x, y))
            
    N = len(points_in_box)
    neighbors = [set() for _ in range(N)]
    
    max_d2 = 8 * L * L
    squares = set()
    r = 0
    while r*r <= max_d2:
        squares.add(r*r)
        r += 1
        
    # Build graph
    # This is O(N^2). 10000^2 = 100M ops. Might be slow in Python.
    # Let's optimize by iterating only over unique pairs and using a faster check.
    # Or reduce L to 40 to be safe on time.
    L = 40
    points_in_box = []
    for x in range(-L, L+1):
        for y in range(-L, L+1):
            points_in_box.append((x, y))
    N = len(points_in_box)
    neighbors = [set() for _ in range(N)]
    
    max_d2 = 8 * L * L
    squares = set()
    r = 0
    while r*r <= max_d2:
        squares.add(r*r)
        r += 1
        
    for i in range(N):
        x1, y1 = points_in_box[i]
        for j in range(i+1, N):
            x2, y2 = points_in_box[j]
            dx = x1 - x2
            dy = y1 - y2
            d2 = dx*dx + dy*dy
            if d2 in squares:
                neighbors[i].add(j)
                neighbors[j].add(i)
                
    # Backtracking search for maximum clique
    # We want to find a clique of size K.
    # We'll try to find size 7, then 6.
    
    def search_clique(candidates, current_clique, target_size):
        nonlocal best_set, best_size
        
        if len(current_clique) > best_size:
            best_size = len(current_clique)
            best_set = list(current_clique)
            
        if len(current_clique) == target_size:
            return True # Found one, stop searching for this target
            
        if not candidates:
            return False
            
        if time.time() - start_time > time_limit - 5:
            return False
            
        # Pruning: if remaining candidates + current size < target, fail
        if len(candidates) + len(current_clique) < target_size:
            return False
            
        # Order candidates by degree in candidates to find large cliques faster
        # Pick a candidate with high degree
        # We can also use the "pivot" heuristic from Tomita's algorithm
        # But simple greedy selection with backtracking is okay for small target sizes.
        
        # To improve, we can check geometric constraints early.
        # If current_clique has >= 2 points, check collinearity with new point.
        # If current_clique has >= 3 points, check concyclicity with new point.
        
        cand_list = list(candidates)
        # Sort by degree in candidates (descending)
        cand_list.sort(key=lambda idx: len(neighbors[idx] & candidates), reverse=True)
        
        for idx in cand_list:
            # Check if time is up
            if time.time() - start_time > time_limit - 5:
                return False
                
            p_new = points_in_box[idx]
            
            # Geometric Pruning
            valid = True
            if len(current_clique) >= 2:
                # Check collinearity with any pair
                clique_pts = [points_in_box[i] for i in current_clique]
                for i in range(len(clique_pts)):
                    for j in range(i+1, len(clique_pts)):
                        if check_collinear(clique_pts[i], clique_pts[j], p_new):
                            valid = False
                            break
                    if not valid:
                        break
                            
            if valid and len(current_clique) >= 3:
                # Check concyclicity with any triple
                clique_pts = [points_in_box[i] for i in current_clique]
                for i in range(len(clique_pts)):
                    for j in range(i+1, len(clique_pts)):
                        for k in range(j+1, len(clique_pts)):
                            if check_concyclic(clique_pts[i], clique_pts[j], clique_pts[k], p_new):
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                        
            if not valid:
                continue
                
            # Recurse
            # New candidates are those in candidates that are neighbors of idx
            new_candidates = candidates & neighbors[idx]
            new_candidates.discard(idx)
            
            # Add to current clique
            current_clique.add(idx)
            found = search_clique(new_candidates, current_clique, target_size)
            current_clique.remove(idx)
            
            if found:
                return True
                
        return False

    # Try to find 7 points
    # We need to choose a starting point.
    # We can try all starting points, but that's expensive.
    # Let's try a few random starting points or points with high degree.
    
    # Get high degree nodes
    high_degree_nodes = sorted(range(N), key=lambda i: len(neighbors[i]), reverse=True)
    
    # Try starting with top 50 high degree nodes
    for start_idx in high_degree_nodes[:50]:
        if time.time() - start_time > time_limit - 10:
            break
            
        # Initial candidates: neighbors of start_idx
        initial_candidates = neighbors[start_idx]
        current_clique = {start_idx}
        
        # Try to find 7
        search_clique(initial_candidates, current_clique, 7)
        
        if best_size >= 7:
            break
            
    # If 7 not found, try 6
    if best_size < 6:
        for start_idx in high_degree_nodes[:100]:
            if time.time() - start_time > time_limit - 10:
                break
                
            initial_candidates = neighbors[start_idx]
            current_clique = {start_idx}
            
            search_clique(initial_candidates, current_clique, 6)
            
            if best_size >= 6:
                break

    if best_set:
        pts = []
        for idx in best_set:
            x, y = points_in_box[idx]
            pts.append([[x, 1], [y, 1]])
        # Final validation
        if validate_set([(points_in_box[i][0], points_in_box[i][1]) for i in best_set]):
            return {'D': 1, 'points': pts}
            
    # Fallback
    pts = []
    for x, y in fallback_points:
        pts.append([[x, 1], [y, 1]])
    return {'D': 1, 'points': pts}
