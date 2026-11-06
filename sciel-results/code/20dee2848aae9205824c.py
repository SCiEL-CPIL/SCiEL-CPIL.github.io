import time
import math
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _line_key(p1, p2):
    """Return canonical line key for two points."""
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    if a == 0 and b == 0:
        return None
    g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
    if g == 0:
        g = 1
    a, b, c = a // g, b // g, c // g
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _count_ordinary_fast(points):
    """Count ordinary lines using a dict of line->count."""
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_counts[key] = line_counts.get(key, 0) + 1
    return sum(1 for v in line_counts.values() if v == 1)

def _is_valid(points):
    """Check: 12 distinct points, not all collinear."""
    if len(points) != 12:
        return False
    if len(set(tuple(p) for p in points)) != 12:
        return False
    (x1, y1), (x2, y2), (x3, y3) = points[0], points[1], points[2]
    if (x2 - x1) * (y3 - y1) != (y2 - y1) * (x3 - x1):
        return True
    for i in range(3, 12):
        (x4, y4) = points[i]
        if (x2 - x1) * (y4 - y1) != (y2 - y1) * (x4 - x1):
            return True
    return False

def _local_search(points, deadline):
    """
    Efficient local search with incremental scoring.
    """
    current = [list(p) for p in points]
    current_score = _count_ordinary_fast(current)
    best = [list(p) for p in current]
    best_score = current_score
    
    # Candidate pool: points on lines defined by pairs of current points, extended
    coords = set()
    for i in range(12):
        for j in range(i+1, 12):
            x1, y1 = current[i]
            x2, y2 = current[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                g = 1
            dx //= g
            dy //= g
            # Extend line
            for t in range(-5, 6):
                coords.add((x1 + t*dx, y1 + t*dy))
    
    # Also add nearby points
    for p in current:
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                coords.add((p[0]+dx, p[1]+dy))
    
    candidate_list = list(coords)
    random.shuffle(candidate_list)
    
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        random.shuffle(candidate_list)
        
        for i in range(12):
            if time.time() >= deadline:
                break
            old_pt = tuple(current[i])
            for cand in candidate_list:
                if cand == old_pt:
                    continue
                if any(cand == tuple(current[j]) for j in range(12) if j != i):
                    continue
                
                new_pts = [list(p) for p in current]
                new_pts[i] = list(cand)
                
                if not _is_valid(new_pts):
                    continue
                
                score = _count_ordinary_fast(new_pts)
                if score < current_score:
                    current = new_pts
                    current_score = score
                    improved = True
                    no_improve = 0
                    if current_score < best_score:
                        best_score = current_score
                        best = [list(p) for p in current]
                    break
            else:
                no_improve += 1
                continue
            break
        else:
            no_improve += 1
            continue
        
        if not improved:
            no_improve += 1
    
    return best, best_score

def construct():
    """Return 12 integer points minimizing ordinary lines."""
    deadline = time.time() + 140
    
    # Start with 4x3 grid
    initial_pts = [[i, j] for i in range(4) for j in range(3)]
    
    best_points, best_score = _local_search(initial_pts, deadline)
    
    # Try a few random restarts if time permits
    if best_score > 6 and time.time() < deadline - 10:
        random.seed(42)
        for restart in range(20):
            if time.time() >= deadline - 5:
                break
            # Random grid-like start
            pts = [[random.randint(-10, 10), random.randint(-10, 10)] for _ in range(12)]
            # Ensure distinct
            while len(set(tuple(p) for p in pts)) < 12:
                pts[random.randint(0, 11)] = [random.randint(-10, 10), random.randint(-10, 10)]
            
            ls_points, ls_score = _local_search(pts, deadline)
            if ls_score < best_score:
                best_score = ls_score
                best_points = ls_points
                
    return [[int(p[0]), int(p[1])] for p in best_points]

# EVOLVE-BLOCK-END
