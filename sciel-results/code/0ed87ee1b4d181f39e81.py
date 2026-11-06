import time
from math import gcd
from itertools import combinations

def _count_ordinary(pts):
    """Count lines containing exactly 2 of the given points."""
    n = len(pts)
    if n < 2:
        return 0
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2
            g = gcd(gcd(abs(a), abs(b)), abs(c))
            if g > 0:
                a //= g
                b //= g
                c //= g
            # Canonical form: first nonzero of (a,b) positive
            if a < 0 or (a == 0 and b < 0):
                a, b, c = -a, -b, -c
            key = (a, b, c)
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    return sum(1 for v in line_map.values() if len(v) == 2)

def _is_valid(pts):
    """Check 12 distinct integer points, |x|,|y| <= 1e9, not all collinear."""
    if len(pts) != 12:
        return False
    seen = set()
    for p in pts:
        if not isinstance(p[0], int) or not isinstance(p[1], int):
            return False
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return False
        key = (p[0], p[1])
        if key in seen:
            return False
        seen.add(key)
    # Check not all collinear
    if len(pts) < 3:
        return True
    x0, y0 = pts[0]
    x1, y1 = pts[1]
    for i in range(2, len(pts)):
        xi, yi = pts[i]
        if (xi - x0) * (y1 - y0) != (yi - y0) * (x1 - x0):
            return True
    return False

def _get_line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    g = gcd(gcd(abs(a), abs(b)), abs(c))
    if g > 0:
        a //= g
        b //= g
        c //= g
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _intersect_lines(k1, k2):
    """
    Returns (x, y) if integer intersection, else None.
    k1, k2 are (a, b, c) tuples.
    """
    a1, b1, c1 = k1
    a2, b2, c2 = k2
    D = a1 * b2 - a2 * b1
    if D == 0:
        return None # Parallel or coincident
    x_num = b1 * c2 - b2 * c1
    y_num = a2 * c1 - a1 * c2
    if x_num % D != 0 or y_num % D != 0:
        return None
    x = x_num // D
    y = y_num // D
    if abs(x) > 10**9 or abs(y) > 10**9:
        return None
    return (x, y)

def _generate_candidates(pts, i):
    """
    Generate candidate positions for point i.
    """
    candidates = set()
    # 1. Local neighborhood
    xi, yi = pts[i]
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            if dx == 0 and dy == 0:
                continue
            candidates.add((xi + dx, yi + dy))
            
    # 2. Intersections of lines formed by other points
    # We only consider lines formed by pairs of points j, k where j,k != i
    other_indices = [j for j in range(12) if j != i]
    # To limit computation, we sample pairs or use all if fast enough
    # 11 points -> 55 pairs. 55*55/2 = 1540 pairs of lines.
    # This is small.
    lines = []
    for j in other_indices:
        for k in other_indices:
            if j >= k:
                continue
            key = _get_line_key(pts[j], pts[k])
            lines.append(key)
            
    # Remove duplicates in lines list to reduce intersection checks
    unique_lines = list(set(lines))
    
    # Intersect all pairs of unique lines
    for u in range(len(unique_lines)):
        for v in range(u + 1, len(unique_lines)):
            pt = _intersect_lines(unique_lines[u], unique_lines[v])
            if pt is not None:
                candidates.add(pt)
                
    return list(candidates)

def construct():
    deadline = time.time() + 140
    
    # Initial configurations
    initial_configs = []
    
    # 3x4 Grid
    initial_configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # 4x3 Grid
    initial_configs.append([[x, y] for x in range(3) for y in range(4)])
    
    # Triangle with points on sides
    # (0,0), (10,0), (0,10)
    # Side y=0: (0,0), (3,0), (6,0), (10,0)
    # Side x=0: (0,0), (0,3), (0,6), (0,10)
    # Side x+y=10: (10,0), (7,3), (3,7), (0,10)
    # Shared vertices: (0,0), (10,0), (0,10)
    # Distinct: (3,0), (6,0), (0,3), (0,6), (7,3), (3,7) -> 6 points + 3 vertices = 9.
    # Add 3 interior: (2,2), (4,4), (2,4)?
    initial_configs.append([
        [0,0], [3,0], [6,0], [10,0],
        [0,3], [0,6], [0,10],
        [7,3], [3,7],
        [2,2], [4,4], [2,4]
    ])
    
    # Random start
    import random
    random.seed(42)
    random_pts = []
    used = set()
    while len(random_pts) < 12:
        x = random.randint(-10, 10)
        y = random.randint(-10, 10)
        if (x, y) not in used:
            used.add((x, y))
            random_pts.append([x, y])
    initial_configs.append(random_pts)
    
    best_pts = None
    best_score = float('inf')
    
    for config in initial_configs:
        if _is_valid(config):
            score = _count_ordinary(config)
            if score < best_score:
                best_score = score
                best_pts = [list(p) for p in config]
                
    if best_pts is None:
        best_pts = initial_configs[0]
        best_score = _count_ordinary(best_pts)
        
    # Local Search
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        improved = False
        for i in range(12):
            if time.time() > deadline:
                break
            candidates = _generate_candidates(best_pts, i)
            for cx, cy in candidates:
                # Check validity
                if abs(cx) > 10**9 or abs(cy) > 10**9:
                    continue
                # Check duplicate
                dup = False
                for j in range(12):
                    if j != i and best_pts[j][0] == cx and best_pts[j][1] == cy:
                        dup = True
                        break
                if dup:
                    continue
                    
                new_pts = [list(p) for p in best_pts]
                new_pts[i] = [cx, cy]
                
                if not _is_valid(new_pts):
                    continue
                    
                score = _count_ordinary(new_pts)
                if score < best_score:
                    best_pts = new_pts
                    best_score = score
                    improved = True
                    break # Move to next point i
                    
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Try a random restart or perturbation
                # Perturb one point randomly
                idx = random.randint(0, 11)
                ox, oy = best_pts[idx]
                best_pts[idx] = [ox + random.randint(-5, 5), oy + random.randint(-5, 5)]
                # Re-evaluate
                if _is_valid(best_pts):
                    best_score = _count_ordinary(best_pts)
                else:
                    # Rollback if invalid? We should ensure validity.
                    best_pts[idx] = [ox, oy]
                no_improve_count = 0
        else:
            no_improve_count = 0
            
    return [list(p) for p in best_pts]
