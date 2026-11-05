import time
import random
from math import gcd

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 3:
        return 0
    # Map line key -> set of point indices
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            g = _gcd(_gcd(abs(A), abs(B)), abs(C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            key = (A, B, C)
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    
    ordinary = 0
    for key, idxs in line_map.items():
        if len(idxs) == 2:
            ordinary += 1
    return ordinary

def _local_search(points, deadline_fn, max_iter=200000):
    """Local search: try moving each point to a candidate position."""
    current = [list(p) for p in points]
    current_score = _count_ordinary_lines(current)
    
    # Build candidate pool: points on lines extending from current configuration
    cand_pool = set()
    for i in range(12):
        for j in range(i + 1, 12):
            x1, y1 = current[i]
            x2, y2 = current[j]
            dx, dy = x2 - x1, y2 - y1
            # Extend in both directions
            for k in [-4, -3, -2, 2, 3, 4]:
                px, py = x1 + k * dx, y1 + k * dy
                if abs(px) <= 10**9 and abs(py) <= 10**9:
                    cand_pool.add((px, py))
    
    # Also add some random candidates
    for _ in range(200):
        px = random.randint(-20, 20)
        py = random.randint(-20, 20)
        cand_pool.add((px, py))
    
    cand_list = list(cand_pool)
    
    for iteration in range(max_iter):
        if deadline_fn() > 115:
            break
        improved = False
        # Shuffle for diversity
        if iteration % 10 == 0:
            random.shuffle(cand_list)
        
        for idx in range(12):
            for cand in cand_list:
                new_pts = [list(p) for p in current]
                new_pts[idx] = list(cand)
                if len(set(map(tuple, new_pts))) != 12:
                    continue
                s = _count_ordinary_lines(new_pts)
                if s < current_score:
                    current = new_pts
                    current_score = s
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return current, current_score

def construct():
    start = time.time()
    deadline = lambda: time.time() - start
    
    best_points = None
    best_score = float('inf')
    
    # Configuration 1: 3x4 grid
    pts = [[x, y] for x in range(3) for y in range(4)]
    s = _count_ordinary_lines(pts)
    if s < best_score:
        best_points = [list(p) for p in pts]
        best_score = s
    
    # Configuration 2: 4x3 grid
    pts = [[x, y] for x in range(4) for y in range(3)]
    s = _count_ordinary_lines(pts)
    if s < best_score:
        best_points = [list(p) for p in pts]
        best_score = s
    
    # Configuration 3: 3x3 grid + 3 carefully chosen
    grid = [[x, y] for x in range(3) for y in range(3)]
    for a, b, c in [((3,0),(0,3),(3,3)), ((3,0),(0,3),(1,3)), 
                     ((3,0),(3,3),(0,3)), ((2,3),(3,2),(3,3)),
                     ((3,1),(1,3),(3,3)), ((3,0),(0,3),(2,3))]:
        pts = grid + [list(a), list(b), list(c)]
        if len(set(map(tuple, pts))) == 12:
            s = _count_ordinary_lines(pts)
            if s < best_score:
                best_points = [list(p) for p in pts]
                best_score = s
    
    # Configuration 4: Two parallel lines with 6 points each, offset
    pts = [[x, 0] for x in range(6)] + [[x, 2] for x in range(6)]
    s = _count_ordinary_lines(pts)
    if s < best_score:
        best_points = [list(p) for p in pts]
        best_score = s
    
    # Configuration 5: Hexagonal-ish arrangement
    pts = [[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[1,2],[2,2],[0,2],[3,0],[3,1],[3,2]]
    s = _count_ordinary_lines(pts)
    if s < best_score:
        best_points = [list(p) for p in pts]
        best_score = s
    
    # Run local search from best
    if best_points is not None and deadline() < 100:
        p, s = _local_search(best_points, deadline, max_iter=150000)
        if s < best_score:
            best_points = p
            best_score = s
    
    # Try local search from a few other configs if time permits
    if deadline() < 90:
        for cfg_name, cfg in [("3x3+extras", grid + [[3,0],[0,3],[3,3]]),
                               ("4x3", [[x,y] for x in range(4) for y in range(3)])]:
            if deadline() > 110:
                break
            p, s = _local_search(cfg, deadline, max_iter=80000)
            if s < best_score:
                best_points = p
                best_score = s
    
    if best_points is None:
        best_points = [[x, y] for x in range(3) for y in range(4)]
    
    return best_points
