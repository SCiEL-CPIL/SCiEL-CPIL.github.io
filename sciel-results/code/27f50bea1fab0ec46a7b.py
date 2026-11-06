import time
import math
import random
from collections import defaultdict
from math import gcd

def count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 3:
        return 0
    lines = defaultdict(set)
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            if dx < 0:
                dx = -dx
                dy = -dy
            elif dx == 0 and dy < 0:
                dy = -dy
            a = dy
            b = -dx
            c = a * x1 + b * y1
            key = (a, b, c)
            lines[key].add(i)
            lines[key].add(j)
    ordinary = 0
    for pts in lines.values():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def _is_valid(pts):
    """Check: 12 distinct integer points, |x|,|y|<=10^9, not all collinear."""
    if len(pts) != 12:
        return False
    if len(set(tuple(p) for p in pts)) != 12:
        return False
    for p in pts:
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return False
    (x0, y0), (x1, y1), (x2, y2) = pts[0], pts[1], pts[2]
    all_col = True
    for i in range(3, 12):
        xi, yi = pts[i]
        if (x1 - x0) * (yi - y0) != (y1 - y0) * (xi - x0):
            all_col = False
            break
    if all_col:
        return False
    return True

def _make_initial_configs():
    """Generate diverse structured starting configurations."""
    configs = []
    
    # 3x4 grid
    configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # 4x3 grid  
    configs.append([[x, y] for x in range(3) for y in range(4)])
    
    # 2x6 grid
    configs.append([[x, y] for x in range(6) for y in range(2)])
    
    # 6x2 grid
    configs.append([[x, y] for x in range(2) for y in range(6)])
    
    # Diagonal-shifted: 4 rows of 3, each shifted
    pts_diag = []
    for y in range(4):
        for x in range(3):
            pts_diag.append([x + y, y])
    configs.append(pts_diag)
    
    # 4x4 minus 4 corners
    pts = [[x, y] for x in range(4) for y in range(4)]
    for corner in [[0, 0], [3, 0], [0, 3], [3, 3]]:
        if corner in pts:
            pts.remove(corner)
    configs.append(pts)
    
    # Hexagonal-like: two staggered rows
    pts_hex = []
    for x in range(6):
        pts_hex.append([2 * x, 0])
    for x in range(6):
        pts_hex.append([2 * x + 1, 2])
    configs.append(pts_hex)
    
    # Three parallel lines of 4 with offsets
    pts_3row = []
    for x in range(4):
        pts_3row.append([x, 0])
    for x in range(1, 5):
        pts_3row.append([x, 2])
    for x in range(2, 6):
        pts_3row.append([x, 4])
    configs.append(pts_3row)
    
    # Triangle-based: points on sides of triangle
    # Bottom: (0,0),(2,0),(4,0),(6,0)
    # Left: (0,0),(1,2),(2,4),(3,6)  
    # Right: (6,0),(5,2),(4,4),(3,6)
    # That's 4+4+4 - 3 shared = 9, add 3 more
    pts_tri = [[0,0],[2,0],[4,0],[6,0],[1,2],[2,4],[3,6],[5,2],[4,4]]
    # Add 3 interior points
    pts_tri.append([3, 2])
    pts_tri.append([2, 1])
    pts_tri.append([4, 1])
    configs.append(pts_tri)
    
    # Pappus-like configuration
    # Two lines with 4 points each, plus 4 intersection points
    # Line 1: y=0, x=0,1,2,3
    # Line 2: y=4, x=1,2,3,4
    # Cross lines create intersections
    pts_pappus = [[0,0],[1,0],[2,0],[3,0],[1,4],[2,4],[3,4],[4,4]]
    # Add 4 more to make 12
    pts_pappus.append([0, 2])
    pts_pappus.append([5, 2])
    pts_pappus.append([2, 1])
    pts_pappus.append([3, 3])
    configs.append(pts_pappus)
    
    # Random configs
    rng = random.Random(12345)
    for _ in range(20):
        pts = []
        for _ in range(12):
            pts.append([rng.randint(-10, 10), rng.randint(-10, 10)])
        if len(set(map(tuple, pts))) == 12:
            configs.append(pts)
    
    # Filter valid configs
    valid_configs = []
    for cfg in configs:
        if _is_valid(cfg):
            valid_configs.append(cfg)
    
    return valid_configs

def _try_move_to_line(points, idx, target_points, max_k=8):
    """Try to move points[idx] onto the line through two other points."""
    best_new = None
    best_score = float('inf')
    
    i, j = target_points
    p1 = points[i]
    p2 = points[j]
    
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return None, float('inf')
        
    g = gcd(abs(dx), abs(dy))
    if g == 0:
        return None, float('inf')
    dx_n = dx // g
    dy_n = dy // g
    
    for k in range(-max_k, max_k + 1):
        if k == 0:
            continue
        nx = p1[0] + k * dx_n
        ny = p1[1] + k * dy_n
        
        if abs(nx) > 10**9 or abs(ny) > 10**9:
            continue
            
        new_p = [nx, ny]
        
        # Check if already occupied
        if any(p == new_p for p in points if points.index(p) != idx):
            continue
            
        new_points = [p[:] for p in points]
        new_points[idx] = new_p
        
        if not _is_valid(new_points):
            continue
            
        score = count_ordinary_lines(new_points)
        if score < best_score:
            best_score = score
            best_new = new_points
    
    return best_new, best_score

def _random_jitter(points, idx, rng, max_delta=5):
    """Randomly perturb a point."""
    old_p = points[idx]
    dx = rng.randint(-max_delta, max_delta)
    dy = rng.randint(-max_delta, max_delta)
    if dx == 0 and dy == 0:
        return None
    
    new_p = [old_p[0] + dx, old_p[1] + dy]
    if abs(new_p[0]) > 10**9 or abs(new_p[1]) > 10**9:
        return None
    
    new_points = [p[:] for p in points]
    new_points[idx] = new_p
    
    if len(set(map(tuple, new_points))) < 12:
        return None
    if not _is_valid(new_points):
        return None
    
    return new_points

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Generate and evaluate initial configs
    initial_configs = _make_initial_configs()
    
    best_points = None
    best_score = float('inf')
    
    for cfg in initial_configs:
        if time.time() > deadline:
            break
        score = count_ordinary_lines(cfg)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in cfg]
        if best_score == 6:
            break
    
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(3)]
        best_score = count_ordinary_lines(best_points)
    
    # Local search
    rng = random.Random(42)
    no_improve_count = 0
    max_no_improve = 50000
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        improved = False
        
        # Strategy 1: Move a point onto a line through two others
        idx = rng.randint(0, 11)
        others = [k for k in range(12) if k != idx]
        i, j = rng.sample(others, 2)
        
        new_pts, new_score = _try_move_to_line(best_points, idx, (i, j), max_k=10)
        if new_pts is not None and new_score < best_score:
            best_points = new_pts
            best_score = new_score
            improved = True
            no_improve_count = 0
            if best_score == 6:
                break
            continue
        
        # Strategy 2: Try multiple pairs for this idx
        if time.time() < deadline:
            for _ in range(3):
                i, j = rng.sample(others, 2)
                new_pts, new_score = _try_move_to_line(best_points, idx, (i, j), max_k=12)
                if new_pts is not None and new_score < best_score:
                    best_points = new_pts
                    best_score = new_score
                    improved = True
                    no_improve_count = 0
                    if best_score == 6:
                        break
                    break
            if best_score == 6:
                break
        
        # Strategy 3: Random jitter
        if time.time() < deadline and not improved:
            idx = rng.randint(0, 11)
            new_pts = _random_jitter(best_points, idx, rng, max_delta=8)
            if new_pts is not None:
                score = count_ordinary_lines(new_pts)
                if score < best_score:
                    best_points = new_pts
                    best_score = score
                    improved = True
                    no_improve_count = 0
                    if best_score == 6:
                        break
        
        if not improved:
            no_improve_count += 1
        
        # Periodic restart: if stuck, try a different starting config
        if no_improve_count % 5000 == 0 and no_improve_count > 0:
            if time.time() > deadline - 20:
                break
            # Try a random config
            cfg = rng.choice(initial_configs)
            score = count_ordinary_lines(cfg)
            if score < best_score:
                best_points = [p[:] for p in cfg]
                best_score = score
                no_improve_count = 0
    
    return best_points
