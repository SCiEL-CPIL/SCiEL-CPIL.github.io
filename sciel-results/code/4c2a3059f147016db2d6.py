import time
import random
from math import gcd
from itertools import combinations

_START = time.time()
_DEADLINE = _START + 110.0

def _count_ordinary(points):
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

def _line_key_from_points(p1, p2):
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

def _line_intersection(l1, l2):
    """Find intersection of two lines given as (A,B,C) with Ax+By+C=0."""
    A1, B1, C1 = l1
    A2, B2, C2 = l2
    det = A1 * B2 - A2 * B1
    if det == 0:
        return None
    x = (B1 * C2 - B2 * C1) / det
    y = (A2 * C1 - A1 * C2) / det
    if abs(x - round(x)) > 1e-9 or abs(y - round(y)) > 1e-9:
        return None
    return (int(round(x)), int(round(y)))

def _count_ordinary_fast(points):
    """Faster count using line grouping."""
    n = len(points)
    line_pts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key_from_points(points[i], points[j])
            if key not in line_pts:
                line_pts[key] = set()
            line_pts[key].add(i)
            line_pts[key].add(j)
    ordinary = 0
    for pts_set in line_pts.values():
        if len(pts_set) == 2:
            ordinary += 1
    return ordinary

def _try_line_configuration(line_multisets, rng):
    """
    Given a list of lines (each with a multiset of how many points),
    try to construct 12 points at line intersections.
    Returns (points, score) or (None, 999) on failure.
    """
    # line_multisets: list of (line_key, num_points)
    # We need to place 12 points total at intersections of these lines
    
    # Get all unique lines
    lines = list(set(l for l, _ in line_multisets))
    if len(lines) < 2:
        return None, 999
    
    # Find all intersection points of pairs of lines
    intersections = {}  # point -> set of line indices passing through it
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            pt = _line_intersection(lines[i], lines[j])
            if pt is not None and abs(pt[0]) <= 10**9 and abs(pt[1]) <= 10**9:
                if pt not in intersections:
                    intersections[pt] = set()
                intersections[pt].add(i)
                intersections[pt].add(j)
    
    if not intersections:
        return None, 999
    
    # Sort intersection points by how many lines pass through them (more constrained first)
    sorted_pts = sorted(intersections.items(), key=lambda x: -len(x[1]))
    
    # Try to select 12 points that satisfy the line constraints
    # Each line must have exactly the specified number of points
    line_target = {}
    for l, count in line_multisets:
        line_target[l] = count
    
    # Map line_key -> index in lines list
    line_idx = {l: i for i, l in enumerate(lines)}
    
    # Greedy: pick points in order of constraint satisfaction
    selected = []
    line_counts = {i: 0 for i in range(len(lines))}
    
    for pt, line_set in sorted_pts:
        if len(selected) >= 12:
            break
        # Check if adding this point doesn't exceed any line's target
        can_add = True
        for li in line_set:
            if line_counts[li] >= line_target.get(lines[li], 0):
                can_add = False
                break
        if can_add:
            # Check if it's needed (some line still needs points)
            needed = False
            for li in line_set:
                if line_counts[li] < line_target.get(lines[li], 0):
                    needed = True
                    break
            if needed:
                selected.append(pt)
                for li in line_set:
                    line_counts[li] += 1
    
    if len(selected) < 12:
        # Fill remaining from other intersections
        for pt, line_set in sorted_pts:
            if len(selected) >= 12:
                break
            if pt in selected:
                continue
            can_add = True
            for li in line_set:
                if line_counts[li] >= line_target.get(lines[li], 0):
                    can_add = False
                    break
            if can_add:
                selected.append(pt)
                for li in line_set:
                    line_counts[li] += 1
    
    if len(selected) < 12:
        return None, 999
    
    # Check all lines have correct count
    for li in range(len(lines)):
        if line_counts[li] != line_target.get(lines[li], 0):
            return None, 999
    
    points = [[pt[0], pt[1]] for pt in selected]
    if not _valid(points):
        return None, 999
    
    score = _count_ordinary_fast(points)
    return points, score

def _systematic_search(rng):
    """
    Systematically try line-based configurations.
    Key insight: to minimize ordinary lines, we want many lines with 3+ points.
    For 12 points, if we have L lines each with k points, total incidences = 12*k_avg.
    Each pair of points determines one line, so C(12,2)=66 pairs.
    A line with m points accounts for C(m,2) pairs.
    Non-ordinary lines save pairs: sum over non-ordinary lines of (C(m,2)-1) = 66 - ordinary_count.
    To get 6 ordinary lines, we need 60 pairs covered by non-ordinary lines.
    """
    best_pts = None
    best_score = 999
    
    # Try various line configurations
    # Each is a list of (num_points_on_line, line_index)
    # We'll try to realize these with actual geometric lines
    
    configurations = [
        # 4 lines of 3 points each: 4*C(3,2)=12 pairs covered, ordinary=66-12=54? No...
        # Actually need to think more carefully.
        # 6 lines of 3 points: 6*3=18 pairs, ordinary = 66-18=48. Still not 6.
        # We need lines with more points or overlapping.
        # 3 lines of 4 points: 3*6=18, ordinary=48
        # 2 lines of 4 + 2 lines of 3: 2*6+2*3=18, ordinary=48
        # 1 line of 5 + ... 
        # Actually the minimum of 6 ordinary lines for 12 points is achieved by specific configs.
        # Let's try: 4 lines with 3 points, arranged so they share points.
        
        # Try: 3 concurrent lines with 4 points each (sharing 1 common point)
        # Line 1: 4 pts, Line 2: 4 pts, Line 3: 4 pts, all sharing 1 point
        # Total unique points: 1 + 3*3 = 10. Need 2 more.
        # Add 2 more points on new lines.
        
        # Try: 4 lines forming a complete quadrilateral + extras
        # Lines: L1(3pts), L2(3pts), L3(3pts), L4(3pts) with various intersections
        
        # Let's just try many random line arrangements
    ]
    
    # Randomized line-based construction
    for attempt in range(200):
        if time.time() > _DEADLINE - 10:
            break
        
        # Choose 4-6 random lines
        num_lines = rng.randint(4, 6)
        lines = []
        for _ in range(num_lines):
            # Random line through small integer points
            x1, y1 = rng.randint(-3, 3), rng.randint(-3, 3)
            x2, y2 = rng.randint(-3, 3), rng.randint(-3, 3)
            if (x1, y1) == (x2, y2):
                continue
            key = _line_key_from_points((x1, y1), (x2, y2))
            if key not in [l for l, _ in lines]:
                lines.append((key, rng.randint(3, 4)))
        
        if len(lines) < 3:
            continue
        
        # Check total points needed
        total_needed = sum(c for _, c in lines)
        # With shared intersection points, actual unique points < total_needed
        # Just try to construct
        
        pts, score = _try_line_configuration(lines, rng)
        if pts is not None and score < best_score:
            best_score = score
            best_pts = pts
    
    return best_pts, best_score

def _local_search(points, max_iters, rng):
    """Fast local search with systematic neighbor generation."""
    cur = [list(p) for p in points]
    cur_score = _count_ordinary_fast(cur)
    best = [list(p) for p in cur]
    best_score = cur_score
    cur_set = set(tuple(p) for p in cur)
    
    # Build a good candidate pool
    pool = set()
    for p in cur:
        for dx in range(-8, 9):
            for dy in range(-8, 9):
                if dx == 0 and dy == 0:
                    continue
                np = (p[0] + dx, p[1] + dy)
                if abs(np[0]) <= 10**9 and abs(np[1]) <= 10**9:
                    pool.add(np)
    # Add line extensions
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
            for k in range(-6, 7):
                if k == 0:
                    continue
                for base in [(x1, y1), (x2, y2)]:
                    nx = base[0] + k * dx
                    ny = base[1] + k * dy
                    if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                        pool.add((nx, ny))
    
    pool = list(pool)
    
    T = 2.0
    cooling = 0.998
    T_min = 0.05
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5:
            break
        
        idx = rng.randrange(12)
        old_pt = tuple(cur[idx])
        
        # Try multiple candidates per iteration
        found_improvement = False
        for _ in range(20):
            cand = tuple(rng.choice(pool))
            if cand == old_pt or cand in cur_set:
                continue
            
            new_pts = [list(p) for p in cur]
            new_pts[idx] = list(cand)
            if not _valid(new_pts):
                continue
            
            new_score = _count_ordinary_fast(new_pts)
            delta = new_score - cur_score
            
            if delta < 0 or (T > T_min and rng.random() < 2.718281828 ** (-delta / T)):
                cur = new_pts
                cur_score = new_score
                cur_set = set(tuple(p) for p in cur)
                found_improvement = True
                if cur_score < best_score:
                    best_score = cur_score
                    best = [list(p) for p in cur]
                break
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best, best_score

def _known_configs():
    configs = []
    # 4x3 grid
    configs.append([[x, y] for x in range(4) for y in range(3)])
    # 3x4 grid  
    configs.append([[x, y] for x in range(3) for y in range(4)])
    # 6x2 grid
    configs.append([[x, y] for x in range(6) for y in range(2)])
    # 2x6 grid
    configs.append([[x, y] for x in range(2) for y in range(6)])
    # 3x3 + 3
    configs.append([[x, y] for x in range(3) for y in range(3)] + [[3,0],[0,3],[3,3]])
    # Near-collinear
    configs.append([[0,0],[1,0],[2,0],[3,1],[4,0],[5,1],[6,0],[7,1],[8,0],[9,1],[10,0],[11,1]])
    # Two clusters
    configs.append([[0,0],[1,0],[0,1],[1,1],[5,5],[6,5],[5,6],[6,6],[10,10],[11,10],[10,11],[11,11]])
    # Circle-like
    configs.append([[3,4],[4,3],[5,0],[0,5],[-3,4],[-4,3],[-5,0],[0,-5],[3,-4],[4,-3],[-5,1],[-1,-5]])
    # Random
    rng = random.Random(42)
    pts = []
    while len(pts) < 12:
        x, y = rng.randint(-5, 5), rng.randint(-5, 5)
        if (x, y) not in pts:
            pts.append([x, y])
    configs.append(pts)
    return configs

def construct():
    rng = random.Random(12345)
    
    best_pts = None
    best_score = 999
    
    # Phase 1: Evaluate known configs
    for cfg in _known_configs():
        if not _valid(cfg):
            continue
        score = _count_ordinary_fast(cfg)
        if score < best_score:
            best_score = score
            best_pts = [list(p) for p in cfg]
    
    if best_pts is None:
        best_pts = [[x, y] for x in range(4) for y in range(3)]
        best_score = _count_ordinary_fast(best_pts)
    
    # Phase 2: Systematic line-based construction
    if time.time() < _DEADLINE - 30:
        sys_pts, sys_score = _systematic_search(rng)
        if sys_pts is not None and sys_score < best_score:
            best_score = sys_score
            best_pts = sys_pts
    
    # Phase 3: Local search from best
    if time.time() < _DEADLINE - 15:
        ls_pts, ls_score = _local_search(best_pts, 300000, rng)
        if ls_score < best_score:
            best_score = ls_score
            best_pts = ls_pts
    
    # Phase 4: Local search from other competitive configs
    for cfg in _known_configs():
        if time.time() > _DEADLINE - 10:
            break
        if not _valid(cfg):
            continue
        cfg_score = _count_ordinary_fast(cfg)
        if cfg_score < best_score + 3:
            ls_pts, ls_score = _local_search(cfg, 50000, rng)
            if ls_score < best_score:
                best_score = ls_score
                best_pts = ls_pts
    
    # Phase 5: Final polish
    if time.time() < _DEADLINE - 5:
        ls_pts, ls_score = _local_search(best_pts, 100000, rng)
        if ls_score < best_score:
            best_score = ls_score
            best_pts = ls_pts
    
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print("Points:", pts)
    print("Ordinary lines:", _count_ordinary(pts))
    print("Distinct:", len(set(tuple(p) for p in pts)))
    print("Max coord:", max(abs(c) for p in pts for c in p))
    print("Collinear:", _collinear_check(pts))
