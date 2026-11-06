"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

def _build_line_map(points):
    """Build a map from line key to set of point indices on that line."""
    n = len(points)
    line_points = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            c = dy * x1 - dx * y1
            line_key = (dx, dy, c)
            line_points[line_key].add(i)
            line_points[line_key].add(j)
    return line_points

def _count_lines_exactly_4(points):
    """Count lines containing exactly 4 of the given points."""
    line_map = _build_line_map(points)
    count = 0
    for indices in line_map.values():
        if len(indices) == 4:
            count += 1
    return count

def _max_collinear(points):
    """Return the maximum number of collinear points."""
    line_map = _build_line_map(points)
    mx = 0
    for indices in line_map.values():
        if len(indices) > mx:
            mx = len(indices)
    return mx

def _valid(points):
    """Check all constraints: 16 distinct points, |x|,|y| <= 1000, no five collinear."""
    if len(points) != 16:
        return False
    if len(set(tuple(p) for p in points)) != 16:
        return False
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
    if _max_collinear(points) >= 5:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    random.seed(42)
    
    # Start with 4x4 grid
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score = _count_lines_exactly_4(best_points)
    
    # Try various initial configurations
    initial_configs = []
    
    # 4x4 grid
    initial_configs.append([[x, y] for x in range(4) for y in range(4)])
    
    # Try different grid spacings
    for sx in [1, 2, 3]:
        for sy in [1, 2, 3]:
            cfg = [[x*sx, y*sy] for x in range(4) for y in range(4)]
            if _valid(cfg):
                initial_configs.append(cfg)
    
    # Affine transformations
    for a in range(-3, 4):
        for b in range(-3, 4):
            for c in range(-3, 4):
                for d in range(-3, 4):
                    det = a*d - b*c
                    if det == 0:
                        continue
                    cfg = [[a*x + b*y, c*x + d*y] for x in range(4) for y in range(4)]
                    if _valid(cfg):
                        initial_configs.append(cfg)
    
    # Random configurations
    for trial in range(150):
        pts = []
        for _ in range(16):
            pts.append([random.randint(-40, 40), random.randint(-40, 40)])
        if _valid(pts):
            initial_configs.append(pts)
    
    # Evaluate initial configs
    for cfg in initial_configs:
        if _valid(cfg):
            score = _count_lines_exactly_4(cfg)
            if score > best_score:
                best_score = score
                best_points = [p[:] for p in cfg]
    
    # Local search: move one point at a time
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(16):
            if time.time() > deadline:
                break
            old_x, old_y = best_points[i]
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = old_x + dx
                    new_y = old_y + dy
                    if abs(new_x) > 1000 or abs(new_y) > 1000:
                        continue
                    candidate = [p[:] for p in best_points]
                    candidate[i] = [new_x, new_y]
                    if _valid(candidate):
                        score = _count_lines_exactly_4(candidate)
                        if score > best_score:
                            best_score = score
                            best_points = candidate
                            improved = True
    
    # Second phase: simulated annealing
    if time.time() < deadline:
        current = [p[:] for p in best_points]
        current_score = best_score
        T = 5.0
        T_min = 0.1
        cooling = 0.995
        
        while T > T_min and time.time() < deadline:
            i = random.randint(0, 15)
            old_x, old_y = current[i]
            new_x = old_x + random.randint(-10, 10)
            new_y = old_y + random.randint(-10, 10)
            
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                T *= cooling
                continue
            
            candidate = [p[:] for p in current]
            candidate[i] = [new_x, new_y]
            
            if _valid(candidate):
                cand_score = _count_lines_exactly_4(candidate)
                delta = cand_score - current_score
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    current = candidate
                    current_score = cand_score
                    if cand_score > best_score:
                        best_score = cand_score
                        best_points = [p[:] for p in current]
            T *= cooling
    
    return best_points
# EVOLVE-BLOCK-END
