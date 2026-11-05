import time
import random
from math import gcd
from itertools import combinations

_START = time.time()
_DEADLINE = _START + 110.0  # safety margin

def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points (n=12)."""
    n = len(points)
    if n < 2:
        return 0
    line_counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A or B or C) else 1
            if g:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            key = (A, B, C)
            line_counts[key] = line_counts.get(key, 0) + 1
    return sum(1 for c in line_counts.values() if c == 1)

def _collinear_check(points):
    if len(points) < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    for p in points[2:]:
        if (p[0] - x0) * (y1 - y0) != (p[1] - y0) * (x1 - x0):
            return False
    return True

def _valid(points):
    if len(points) != 12:
        return False
    seen = set()
    for x, y in points:
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        t = (x, y)
        if t in seen:
            return False
        seen.add(t)
    return not _collinear_check(points)

def _line_through_two(p1, p2):
    """Get normalized line coefficients (A, B, C) for line through p1, p2."""
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A or B or C) else 1
    if g:
        A //= g
        B //= g
        C //= g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _local_search_sa(points, max_iters=200000, start_temp=10.0, end_temp=0.1):
    """Simulated annealing local search to minimize ordinary lines."""
    cur = [p[:] for p in points]
    cur_score = _count_ordinary(cur)
    best = [p[:] for p in cur]
    best_score = cur_score
    
    # Build a fixed candidate pool once: neighborhood + line extensions
    pool = set()
    for p in cur:
        for dx in range(-8, 9):
            for dy in range(-8, 9):
                if dx == 0 and dy == 0:
                    continue
                pool.add((p[0] + dx, p[1] + dy))
    # Extend lines through pairs
    for i in range(12):
        for j in range(i + 1, 12):
            x1, y1 = cur[i]
            x2, y2 = cur[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy)) if (dx or dy) else 1
            if g:
                dx //= g
                dy //= g
            for k in range(-4, 5):
                if k == 0:
                    continue
                pool.add((x1 + k*dx, y1 + k*dy))
                pool.add((x2 + k*dx, y2 + k*dy))
    pool_list = list(pool)
    
    iters = 0
    rng = random.Random(42)
    cur_set = set(tuple(p) for p in cur)
    
    while iters < max_iters and time.time() < _DEADLINE - 5:
        iters += 1
        temp = start_temp * (end_temp / start_temp) ** (iters / max_iters)
        
        idx = rng.randint(0, 11)
        old = cur[idx]
        
        # Try several random candidates
        n_try = min(10, len(pool_list))
        candidates = rng.sample(pool_list, n_try)
        improved = False
        
        for cand in candidates:
            if cand in cur_set:
                continue
            new_pts = [p[:] for p in cur]
            new_pts[idx] = list(cand)
            if not _valid(new_pts):
                continue
            new_score = _count_ordinary(new_pts)
            
            delta = new_score - cur_score
            if delta < 0 or (temp > 0 and rng.random() < min(1.0, 1.0 / (1.0 + delta / temp))):
                cur = new_pts
                cur_score = new_score
                cur_set = set(tuple(p) for p in cur)
                improved = True
                if cur_score < best_score:
                    best = [p[:] for p in cur]
                    best_score = cur_score
                break
    
    return best, best_score

def _try_known_configs():
    """Try known good configurations and their variants."""
    configs = []
    
    # 4x3 grid
    configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # 3x4 grid
    configs.append([[x, y] for x in range(3) for y in range(4)])
    
    # 2x6 grid
    configs.append([[x, y] for x in range(2) for y in range(6)])
    
    # 6x2 grid
    configs.append([[x, y] for x in range(6) for y in range(2)])
    
    # Cubic curve points: (t, t^2) for t in 0..11
    configs.append([[t, t*t] for t in range(12)])
    
    # Parabola: (t, t^2) for t in -5..4
    configs.append([[t, t*t] for t in range(-5, 5)])
    
    # 3x3 grid + 3 extra points
    base = [[x, y] for x in range(3) for y in range(3)]
    configs.append(base + [[3, 0], [3, 1], [3, 2]])
    configs.append(base + [[0, 3], [1, 3], [2, 3]])
    
    # 2x3x2 pattern
    configs.append([[x, y] for x in range(2) for y in range(3)] + [[x+3, y] for x in range(2) for y in range(3)])
    
    # Points on two lines: 6 on each, but not all collinear
    configs.append([[x, 0] for x in range(6)] + [[x, 1] for x in range(6)])
    
    # 3 lines with 4 points each
    configs.append([[0, 0], [1, 0], [2, 0], [3, 0],
                    [0, 1], [0, 2], [0, 3],
                    [1, 1], [2, 2], [3, 3],
                    [4, 4], [5, 5]])
    
    # New: 4 lines of 3 points (targeting 6 ordinary lines)
    # Configuration: 3 parallel lines with 4 points each won't work (too many ordinary)
    # Try: 4 lines, each with 3 points, arranged so that intersections create more collinearities
    # Example: points at vertices and edge-midpoints of a quadrilateral
    configs.append([
        [0, 0], [1, 0], [2, 0],  # line 1: y=0
        [0, 1], [1, 1], [2, 1],  # line 2: y=1
        [0, 2], [1, 2], [2, 2],  # line 3: y=2
        [0, 3], [1, 3], [2, 3]   # line 4: y=3
    ])
    
    # 4x3 grid is already covered, but let's add a 3x4 variant with better structure
    configs.append([
        [0, 0], [1, 0], [2, 0],
        [0, 1], [1, 1], [2, 1],
        [0, 2], [1, 2], [2, 2],
        [0, 3], [1, 3], [2, 3]
    ])
    
    return configs

def _line_partition_search(points, target_ordinary=6):
    """
    Given a set of 12 points, analyze their line structure.
    Return the number of ordinary lines (lines with exactly 2 points).
    This is used to evaluate candidate configurations.
    """
    return _count_ordinary(points)

def _generate_line_intersection_config():
    """
    Generate a configuration by placing points at line intersections.
    Strategy: Start with 4 lines, place 3 points on each line.
    The intersections of these lines may create additional collinearities.
    
    For 12 points with 4 lines of 3 points each:
    - Each pair of lines intersects at 1 point
    - 4 lines have C(4,2) = 6 intersection points
    - If we place points at these intersections, we get 6 points that are each on 2 lines
    - The remaining 6 points can be placed on the lines away from intersections
    
    This creates a rich collinearity structure.
    """
    configs = []
    
    # Configuration 1: 4 lines in general position (no 3 concurrent)
    # Lines: L1: y=0, L2: y=1, L3: x=0, L4: x=1
    # Intersections: (0,0), (1,0), (0,1), (1,1)
    # Place 3 points on each line:
    # L1 (y=0): (0,0), (1,0), (2,0)
    # L2 (y=1): (0,1), (1,1), (2,1)
    # L3 (x=0): (0,0), (0,1), (0,2)
    # L4 (x=1): (1,0), (1,1), (1,2)
    # But this has duplicates. Let's use distinct points.
    
    # Better: Use 4 lines that don't have parallel pairs
    # L1: y = 0
    # L2: y = 1  
    # L3: x = 0
    # L4: x = 2
    # Intersections: (0,0), (2,0), (0,1), (2,1)
    # Points on L1: (0,0), (1,0), (2,0)
    # Points on L2: (0,1), (1,1), (2,1)
    # Points on L3: (0,0), (0,1), (0,2)
    # Points on L4: (2,0), (2,1), (2,2)
    # Total unique: (0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (2,2) = 8 points
    # Need 12, so this doesn't work directly.
    
    # Let's try a different approach: 3 lines with 4 points each
    # L1: y=0: (0,0), (1,0), (2,0), (3,0)
    # L2: y=1: (0,1), (1,1), (2,1), (3,1)
    # L3: x=1: (1,0), (1,1), (1,2), (1,3)
    # But (1,0) and (1,1) are shared.
    # Unique: (0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1), (1,2), (1,3) = 10 points
    # Still need 2 more.
    
    # Let's just try the 4x3 grid and see what we get, then refine.
    # Actually, let's try a completely different structural approach.
    
    # Configuration: Points on a conic (ellipse) with additional structure
    # 6 points on x-axis, 6 points on a parabola
    configs.append([
        [0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0],  # x-axis
        [0, 1], [1, 4], [2, 9], [3, 16], [4, 25], [5, 36]  # parabola y=x^2
    ])
    
    # Configuration: 2x6 grid with diagonal structure
    configs.append([
        [0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
        [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5]
    ])
    
    # Configuration: 3x4 grid
    configs.append([
        [0, 0], [1, 0], [2, 0],
        [0, 1], [1, 1], [2, 1],
        [0, 2], [1, 2], [2, 2],
        [0, 3], [1, 3], [2, 3]
    ])
    
    # Configuration: 4x3 grid
    configs.append([
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0, 1], [1, 1], [2, 1], [3, 1],
        [0, 2], [1, 2], [2, 2], [3, 2]
    ])
    
    # Configuration: Hexagonal-like structure
    # 6 points in a hexagon, 6 points at center region
    configs.append([
        [0, 0], [2, 0], [3, 1], [2, 2], [0, 2], [-1, 1],  # outer hexagon
        [1, 0], [1, 1], [1, 2], [0, 1], [2, 1], [1, 3]     # inner points
    ])
    
    return configs

def _refine_with_local_search(points, max_iters=50000):
    """
    Refine a configuration using a focused local search.
    Instead of random moves, try specific structural moves:
    1. Move a point to create a new 3-point line
    2. Swap two points to improve line structure
    """
    cur = [p[:] for p in points]
    cur_score = _count_ordinary(cur)
    best = [p[:] for p in cur]
    best_score = cur_score
    
    rng = random.Random(42)
    cur_set = set(tuple(p) for p in cur)
    
    # Build a smarter pool: focus on points that would create 3-point lines
    pool = set()
    for i in range(12):
        for j in range(i + 1, 12):
            x1, y1 = cur[i]
            x2, y2 = cur[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy)) if (dx or dy) else 1
            if g:
                dx //= g
                dy //= g
            # Points on the line through i,j that would create a 3-point line
            for k in range(-3, 4):
                if k == 0 or k == 1:
                    continue
                cand = (x1 + k*dx, y1 + k*dy)
                if abs(cand[0]) <= 10**9 and abs(cand[1]) <= 10**9:
                    pool.add(cand)
    
    # Also add nearby points
    for p in cur:
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                pool.add((p[0] + dx, p[1] + dy))
    
    pool_list = list(pool)
    if not pool_list:
        return best, best_score
    
    iters = 0
    while iters < max_iters and time.time() < _DEADLINE - 5:
        iters += 1
        
        idx = rng.randint(0, 11)
        old = cur[idx]
        
        # Try a few candidates
        n_try = min(5, len(pool_list))
        candidates = rng.sample(pool_list, n_try)
        
        for cand in candidates:
            if cand in cur_set:
                continue
            new_pts = [p[:] for p in cur]
            new_pts[idx] = list(cand)
            if not _valid(new_pts):
                continue
            new_score = _count_ordinary(new_pts)
            
            if new_score < cur_score:
                cur = new_pts
                cur_score = new_score
                cur_set = set(tuple(p) for p in cur)
                if cur_score < best_score:
                    best = [p[:] for p in cur]
                    best_score = cur_score
                break
    
    return best, best_score

def construct():
    best_pts = None
    best_score = float('inf')
    
    # Phase 1: Try known configurations
    all_configs = _try_known_configs() + _generate_line_intersection_config()
    for cfg in all_configs:
        if not _valid(cfg):
            continue
        score = _count_ordinary(cfg)
        if score < best_score:
            best_score = score
            best_pts = [p[:] for p in cfg]
    
    # If we haven't found a good config, use 4x3 grid as fallback
    if best_pts is None:
        best_pts = [[x, y] for x in range(4) for y in range(3)]
        best_score = _count_ordinary(best_pts)
    
    # Phase 2: Refine with focused local search
    if time.time() < _DEADLINE - 10:
        pts, score = _refine_with_local_search(best_pts, max_iters=100000)
        if score < best_score:
            best_pts = pts
            best_score = score
    
    # Phase 3: Try refining other good starting points
    if time.time() < _DEADLINE - 15:
        # Sort configs by their initial score
        scored_configs = []
        for cfg in all_configs:
            if not _valid(cfg):
                continue
            s = _count_ordinary(cfg)
            scored_configs.append((s, cfg))
        scored_configs.sort()
        
        for s, cfg in scored_configs[:6]:
            if time.time() >= _DEADLINE - 10:
                break
            pts, score = _refine_with_local_search(cfg, max_iters=50000)
            if score < best_score:
                best_pts = pts
                best_score = score
    
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print("Points:", pts)
    print("Ordinary lines:", _count_ordinary(pts))
    print("Distinct:", len(set(tuple(p) for p in pts)))
    print("Max coord:", max(abs(c) for p in pts for c in p))
    print("Collinear:", _collinear_check(pts))
