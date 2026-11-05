"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations
from math import gcd

_start_time = time.time()
_DEADLINE = 140  # seconds, safety margin for 150s kill

def _line_key(p1, p2):
    """Compute canonical line key for two points."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    g = gcd(abs(dx), abs(dy))
    if g > 0:
        dx //= g
        dy //= g
    a, b = dy, -dx
    c = a * p1[0] + b * p1[1]
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _count_four_lines(points):
    """Count lines containing exactly 4 of the given points using canonical line signatures."""
    n = len(points)
    line_map = {}
    for i in range(n):
        for j in range(i+1, n):
            key = _line_key(points[i], points[j])
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    
    count = 0
    for key, pts in line_map.items():
        if len(pts) == 4:
            count += 1
    return count

def _is_valid(points):
    """Check that all points are distinct, |x|,|y| <= 1000, and no five collinear."""
    if len(points) != 16:
        return False
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
    if len(set(tuple(p) for p in points)) != 16:
        return False
    line_map = {}
    n = 16
    for i in range(n):
        for j in range(i+1, n):
            key = _line_key(points[i], points[j])
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    for key, pts in line_map.items():
        if len(pts) >= 5:
            return False
    return True

def _affine_transform(points, a, b, c, d, e, f):
    """Apply affine transform: (x,y) -> (ax+by+e, cx+dy+f)"""
    result = []
    for x, y in points:
        nx = a*x + b*y + e
        ny = c*x + d*y + f
        result.append([nx, ny])
    return result

def _initial_grid():
    return [[x, y] for x in range(4) for y in range(4)]

def _try_move_fast(points, idx, dx, dy):
    """Try moving point idx by (dx, dy). Only check lines through the moved point for 5-collinear."""
    new_pts = [list(p) for p in points]
    new_pts[idx] = [new_pts[idx][0] + dx, new_pts[idx][1] + dy]
    x, y = new_pts[idx]
    if abs(x) > 1000 or abs(y) > 1000:
        return None
    for j in range(16):
        if j != idx and new_pts[j] == new_pts[idx]:
            return None
    n = 16
    checked_lines = set()
    for j in range(n):
        if j == idx:
            continue
        key = _line_key(new_pts[idx], new_pts[j])
        if key in checked_lines:
            continue
        checked_lines.add(key)
        a, b, c = key
        cnt = 0
        for k in range(n):
            px, py = new_pts[k]
            if a * px + b * py == c:
                cnt += 1
                if cnt >= 5:
                    return None
    return new_pts

def _local_search(points, time_budget=10.0):
    """Run local search / simulated annealing on a starting configuration."""
    start = time.time()
    current = [list(p) for p in points]
    current_score = _count_four_lines(current)
    best = [list(p) for p in current]
    best_score = current_score
    
    T = 8.0
    T_min = 0.01
    cooling_rate = 0.999
    
    while time.time() - start < time_budget and T > T_min:
        idx = random.randint(0, 15)
        r = random.random()
        if r < 0.5:
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
        elif r < 0.8:
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
        else:
            dx = random.randint(-15, 15)
            dy = random.randint(-15, 15)
        if dx == 0 and dy == 0:
            T *= cooling_rate
            continue
            
        new_pts = _try_move_fast(current, idx, dx, dy)
        if new_pts is None:
            T *= cooling_rate
            continue
            
        new_score = _count_four_lines(new_pts)
        
        delta = new_score - current_score
        if delta > 0:
            current = new_pts
            current_score = new_score
            if current_score > best_score:
                best = [list(p) for p in current]
                best_score = current_score
        else:
            p = math.exp(delta / T)
            if random.random() < p:
                current = new_pts
                current_score = new_score
        
        T *= cooling_rate
        
        # Occasional larger perturbation
        if random.random() < 0.03:
            for _ in range(5):
                idx2 = random.randint(0, 15)
                dx2 = random.randint(-25, 25)
                dy2 = random.randint(-25, 25)
                new_pts2 = _try_move_fast(current, idx2, dx2, dy2)
                if new_pts2 is not None:
                    ns = _count_four_lines(new_pts2)
                    if ns > current_score:
                        current = new_pts2
                        current_score = ns
                        if current_score > best_score:
                            best = [list(p) for p in current]
                            best_score = current_score
                        break
                        
    return best, best_score

def _generate_random_start():
    """Generate a random valid starting configuration."""
    while True:
        pts = []
        for _ in range(16):
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            pts.append([x, y])
        if len(set(tuple(p) for p in pts)) == 16:
            if _is_valid(pts):
                return pts

def construct():
    global _start_time
    random.seed(42)
    
    base_grid = _initial_grid()
    
    # Generate several affine transforms of the base grid
    transforms = [
        (1, 0, 0, 1, 0, 0),
        (2, 0, 0, 2, 0, 0),
        (1, 1, 0, 1, 0, 0),
        (2, 1, 1, 2, 0, 0),
        (1, 0, 0, 2, 0, 0),
        (2, 0, 0, 1, 0, 0),
        (1, -1, 0, 1, 0, 0),
        (3, 0, 0, 3, 0, 0),
        (1, 2, 2, 1, 0, 0),
        (2, 1, -1, 2, 0, 0),
    ]
    
    # Also add some random starts
    random_starts = []
    for _ in range(5):
        rs = _generate_random_start()
        random_starts.append(rs)
    
    all_starts = []
    for a, b, c, d, e, f in transforms:
        transformed = _affine_transform(base_grid, a, b, c, d, e, f)
        if _is_valid(transformed):
            all_starts.append(transformed)
    all_starts.extend(random_starts)
    
    best_overall = None
    best_score_overall = 0
    
    elapsed = time.time() - _start_time
    remaining = _DEADLINE - elapsed
    n_starts = min(len(all_starts), max(1, int(remaining / 10)))
    
    for s_idx in range(n_starts):
        start_pts = all_starts[s_idx]
        
        time_budget = min(10.0, _DEADLINE - (time.time() - _start_time))
        if time_budget < 2.0:
            break
            
        local_best, local_score = _local_search(start_pts, time_budget)
        
        if local_score > best_score_overall:
            best_score_overall = local_score
            best_overall = local_best
    
    if best_overall is None:
        best_overall = base_grid
        best_score_overall = _count_four_lines(base_grid)
    
    # Final polish
    elapsed = time.time() - _start_time
    remaining = _DEADLINE - elapsed
    if remaining > 5.0:
        polished, polished_score = _local_search(best_overall, remaining - 2.0)
        if polished_score > best_score_overall:
            best_overall = polished
            best_score_overall = polished_score
    
    return [[int(x), int(y)] for x, y in best_overall]

# EVOLVE-BLOCK-END
