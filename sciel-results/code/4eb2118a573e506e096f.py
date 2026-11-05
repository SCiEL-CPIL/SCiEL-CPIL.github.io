import time
import random
from math import gcd
from itertools import combinations

_START = time.time()
_DEADLINE = _START + 140.0  # safety margin

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

def _count_ordinary_fast(points):
    """Faster count using integer arithmetic only."""
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

def _local_search_sa(points, max_iters=200000, start_temp=10.0, end_temp=0.1, seed=42):
    """Simulated annealing local search to minimize ordinary lines."""
    cur = [p[:] for p in points]
    cur_score = _count_ordinary(cur)
    best = [p[:] for p in cur]
    best_score = cur_score
    
    def get_pool(pts):
        pool = set()
        for p in pts:
            for dx in range(-6, 7):
                for dy in range(-6, 7):
                    if dx == 0 and dy == 0:
                        continue
                    pool.add((p[0] + dx, p[1] + dy))
        # Extend lines through pairs
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
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
        return list(pool)
    
    pool = get_pool(cur)
    iters = 0
    rng = random.Random(seed)
    cur_set = set(tuple(p) for p in cur)
    
    while iters < max_iters and time.time() < _DEADLINE - 5:
        iters += 1
        temp = start_temp * (end_temp / start_temp) ** (iters / max_iters)
        
        idx = rng.randint(0, 11)
        old = cur[idx]
        
        # Try several random candidates
        n_try = min(12, len(pool))
        candidates = rng.sample(pool, n_try)
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
        
        if not improved and iters % 300 == 0:
            pool = get_pool(cur)
        
        # Periodically try moving two points simultaneously
        if iters % 100 == 0:
            idx1 = rng.randint(0, 11)
            idx2 = rng.randint(0, 11)
            if idx1 == idx2:
                idx2 = (idx2 + 1) % 12
            cand1 = rng.choice(pool)
            cand2 = rng.choice(pool)
            if cand1 not in cur_set and cand2 not in cur_set and cand1 != cand2:
                new_pts = [p[:] for p in cur]
                new_pts[idx1] = list(cand1)
                new_pts[idx2] = list(cand2)
                if _valid(new_pts):
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
    configs.append(base + [[3, 0], [3, 1], [3, 2]])  # 4x3 grid
    configs.append(base + [[0, 3], [1, 3], [2, 3]])  # 3x4 grid
    
    # 2x3x2 pattern
    configs.append([[x, y] for x in range(2) for y in range(3)] + [[x+3, y] for x in range(2) for y in range(3)])
    
    # Points on two lines: 6 on each
    configs.append([[x, 0] for x in range(6)] + [[x, 1] for x in range(6)])
    
    # 3 lines with 4 points each
    configs.append([[0, 0], [1, 0], [2, 0], [3, 0],
                    [0, 1], [0, 2], [0, 3],
                    [1, 1], [2, 2], [3, 3],
                    [4, 4], [5, 5]])
    
    # Pappus-like configuration
    # 3 points on each of 3 lines, plus 3 intersection points
    # Line 1: y=0, Line 2: y=1, Line 3: x=0
    configs.append([[0, 0], [1, 0], [2, 0],
                    [0, 1], [1, 1], [2, 1],
                    [0, 2], [0, 3], [0, 4],
                    [3, 0], [3, 1], [3, 2]])
    
    # Finite field inspired: points from F_3 x F_3 plus some
    configs.append([[x, y] for x in range(3) for y in range(3)] + [[3, 0], [3, 1], [3, 2], [4, 0], [4, 1], [4, 2]])
    
    # Try configurations based on regular polygon approximations
    import math
    for r in [3, 4, 5]:
        pts = []
        for i in range(12):
            angle = 2 * math.pi * i / 12
            x = round(r * math.cos(angle))
            y = round(r * math.sin(angle))
            pts.append([x, y])
        # Ensure distinctness
        seen = set()
        unique_pts = []
        for p in pts:
            if tuple(p) not in seen:
                seen.add(tuple(p))
                unique_pts.append(p)
        if len(unique_pts) == 12:
            configs.append(unique_pts)
    
    # Hexagonal lattice
    configs.append([[0, 0], [1, 0], [2, 0],
                    [0, 1], [1, 1], [2, 1],
                    [0, 2], [1, 2], [2, 2],
                    [0, 3], [1, 3], [2, 3]])
    
    # More diverse configurations
    configs.append([[0, 0], [1, 0], [2, 0], [3, 0],
                    [0, 1], [1, 1], [2, 1], [3, 1],
                    [0, 2], [1, 2], [2, 2], [3, 2]])
    
    return configs

def _backtrack_line_assignment():
    """
    Try to construct a configuration with exactly 6 ordinary lines by
    assigning points to lines such that each line has at least 3 points.
    
    We'll try specific line arrangements that are known to minimize ordinary lines.
    """
    configs = []
    
    # Configuration: 4 lines of 3 points each, with intersections
    # This is the "near-pencil" dual: 4 lines, each pair intersects at a unique point
    # But we need 12 points total.
    
    # Let's try: 3 parallel lines with 4 points each
    configs.append([[x, 0] for x in range(4)] + [[x, 1] for x in range(4)] + [[x, 2] for x in range(4)])
    
    # 4 parallel lines with 3 points each
    configs.append([[x, 0] for x in range(3)] + [[x, 1] for x in range(3)] + 
                   [[x, 2] for x in range(3)] + [[x, 3] for x in range(3)])
    
    # 2 vertical + 2 horizontal lines (grid-like but sparse)
    configs.append([[0, 0], [0, 1], [0, 2], [0, 3],
                    [1, 0], [1, 1], [1, 2], [1, 3],
                    [2, 0], [2, 1], [2, 2], [2, 3]])
    
    # Configuration with more diagonal lines
    configs.append([[0, 0], [1, 1], [2, 2], [3, 3],
                    [0, 1], [1, 0], [2, 3], [3, 2],
                    [0, 2], [1, 3], [2, 0], [3, 1]])
    
    # Try to find configurations with exactly 6 ordinary lines
    # by systematic search over small grids
    for w in range(2, 6):
        for h in range(2, 6):
            if w * h == 12:
                configs.append([[x, y] for x in range(w) for y in range(h)])
    
    # Non-rectangular configurations
    configs.append([[0, 0], [1, 0], [2, 0], [3, 0],
                    [0, 1], [1, 1], [2, 1],
                    [0, 2], [1, 2], [2, 2],
                    [0, 3], [1, 3]])
    
    configs.append([[0, 0], [1, 0], [2, 0],
                    [0, 1], [1, 1], [2, 1], [3, 1],
                    [0, 2], [1, 2], [2, 2], [3, 2],
                    [0, 3], [1, 3]])
    
    return configs

def construct():
    best_pts = None
    best_score = float('inf')
    
    # Try known configurations
    all_configs = _try_known_configs() + _backtrack_line_assignment()
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
    
    # Run simulated annealing from best config
    if time.time() < _DEADLINE - 10:
        pts, score = _local_search_sa(best_pts, max_iters=200000, seed=42)
        if score < best_score:
            best_pts = pts
            best_score = score
    
    # Run SA from other good starting points if time allows
    if time.time() < _DEADLINE - 15:
        for i, cfg in enumerate(all_configs[:10]):
            if not _valid(cfg):
                continue
            if time.time() >= _DEADLINE - 10:
                break
            pts, score = _local_search_sa(cfg, max_iters=50000, seed=100 + i)
            if score < best_score:
                best_pts = pts
                best_score = score
    
    # Final polish: try more iterations if time allows
    if time.time() < _DEADLINE - 5 and best_pts is not None:
        pts, score = _local_search_sa(best_pts, max_iters=100000, start_temp=5.0, end_temp=0.01, seed=999)
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
