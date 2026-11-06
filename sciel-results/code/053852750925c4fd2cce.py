"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time
from itertools import combinations

def _count_ordinary(points):
    """Count lines containing exactly two of the given points."""
    n = len(points)
    if n < 3:
        return 0
    lines = {}
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            # Normalize sign
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            # Line: (a,b,c) where a*x + b*y = c, a = dy, b = -dx
            a, b = dy, -dx
            c = a * xi + b * yi
            key = (a, b, c)
            if key in lines:
                lines[key].add(i)
            else:
                lines[key] = {i, j}
    ordinary = 0
    for pts in lines.values():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def _valid(points):
    if len(points) != 12:
        return False
    if len(set(tuple(p) for p in points)) != 12:
        return False
    if any(abs(p[0]) > 10**9 or abs(p[1]) > 10**9 for p in points):
        return False
    # Not all collinear
    if len(points) >= 3:
        (x0, y0), (x1, y1), (x2, y2) = points[0], points[1], points[2]
        if (x1 - x0) * (y2 - y0) == (y1 - y0) * (x2 - x0):
            # Check all
            collinear = True
            for i in range(3, 12):
                xi, yi = points[i]
                if (x1 - x0) * (yi - y0) != (y1 - y0) * (xi - x0):
                    collinear = False
                    break
            if collinear:
                return False
    return True

def _local_search(points, max_iter, rng):
    """Greedy hill-climbing: move one point to a random nearby integer location."""
    pts = [list(p) for p in points]
    cur_ordinary = _count_ordinary(pts)
    for _ in range(max_iter):
        idx = rng.randrange(12)
        old = pts[idx][:]
        # Try a random displacement
        for _ in range(30):
            dx = rng.randint(-50, 50)
            dy = rng.randint(-50, 50)
            if dx == 0 and dy == 0:
                continue
            nx, ny = old[0] + dx, old[1] + dy
            if abs(nx) > 10**9 or abs(ny) > 10**9:
                continue
            pts[idx] = [nx, ny]
            if len(set(tuple(p) for p in pts)) != 12:
                pts[idx] = old
                continue
            new_ordinary = _count_ordinary(pts)
            if new_ordinary < cur_ordinary:
                cur_ordinary = new_ordinary
                break
            else:
                pts[idx] = old
        if cur_ordinary <= 6:
            break
    return pts, cur_ordinary

def _make_grid_config(rows, cols, rng):
    """Create a grid-like configuration with some randomization."""
    pts = []
    for i in range(rows):
        for j in range(cols):
            x = j * 10 + rng.randint(-2, 2)
            y = i * 10 + rng.randint(-2, 2)
            pts.append([x, y])
    return pts

def _make_known_good():
    """Start from a configuration known to have fewer ordinary lines."""
    # 3x4 grid as baseline
    return [[x, y] for x in range(4) for y in range(3)]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    rng = random.Random(42)
    
    best_pts = None
    best_ordinary = 999
    
    # Start from several initial configurations
    initial_configs = []
    
    # 1. 3x4 grid
    initial_configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # 2. 4x3 grid (transposed)
    initial_configs.append([[y, x] for x in range(4) for y in range(3)])
    
    # 3. Random points in a box
    for trial in range(5):
        pts = []
        for _ in range(12):
            pts.append([rng.randint(-20, 20), rng.randint(-20, 20)])
        if len(set(tuple(p) for p in pts)) == 12:
            initial_configs.append(pts)
    
    # 4. Structured: 3 lines with 4 points each (near-collinear groups)
    # Group 1: y=0, x = 0,1,2,3
    # Group 2: y=10, x = 0,1,2,3  
    # Group 3: x=0, y = 0,10,20,30 -- but need 12 unique
    # Let's do: 4 points on y=0, 4 on y=10, 4 on y=20 but offset
    cfg = []
    for i in range(4):
        cfg.append([i, 0])
    for i in range(4):
        cfg.append([i+1, 10])
    for i in range(4):
        cfg.append([i+2, 20])
    initial_configs.append(cfg)
    
    # 5. Another structured config
    cfg2 = []
    for i in range(4):
        cfg2.append([i, 0])
    for i in range(4):
        cfg2.append([0, i+1])
    for i in range(4):
        cfg2.append([i+1, i+1])
    initial_configs.append(cfg2)
    
    # Evaluate initial configs
    for cfg in initial_configs:
        if not _valid(cfg):
            continue
        ord_count = _count_ordinary(cfg)
        if ord_count < best_ordinary:
            best_ordinary = ord_count
            best_pts = [list(p) for p in cfg]
        if best_ordinary <= 6:
            break
    
    # Local search from best and from multiple restarts
    search_configs = [best_pts] if best_pts else []
    for cfg in initial_configs:
        if _valid(cfg):
            search_configs.append(cfg)
    
    max_iter_per_search = 5000
    total_searches = 0
    
    for cfg in search_configs:
        if time.time() > deadline - 5:
            break
        if not _valid(cfg):
            continue
        pts, ordinary = _local_search(cfg, max_iter_per_search, rng)
        total_searches += 1
        if ordinary < best_ordinary:
            best_ordinary = ordinary
            best_pts = pts
            if best_ordinary <= 6:
                break
    
    # More random restarts if time remains
    while time.time() < deadline - 10:
        # Random initial config
        pts = []
        for _ in range(12):
            pts.append([rng.randint(-30, 30), rng.randint(-30, 30)])
        if len(set(tuple(p) for p in pts)) != 12:
            continue
        if not _valid(pts):
            continue
        pts, ordinary = _local_search(pts, max_iter_per_search, rng)
        total_searches += 1
        if ordinary < best_ordinary:
            best_ordinary = ordinary
            best_pts = pts
            if best_ordinary <= 6:
                break
    
    if best_pts is None:
        # Fallback: return the 3x4 grid
        best_pts = [[x, y] for x in range(4) for y in range(3)]
    
    # Ensure validity
    if not _valid(best_pts):
        best_pts = [[x, y] for x in range(4) for y in range(3)]
    
    return best_pts
# EVOLVE-BLOCK-END
