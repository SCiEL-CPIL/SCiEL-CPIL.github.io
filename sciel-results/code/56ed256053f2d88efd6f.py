import time
import random
import math
from math import gcd

DEADLINE = 145.0  # Safety margin before 150s kill

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    a = -dy
    b = dx
    c = dx * y1 - dy * x1
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
        c = -c
    g2 = gcd(abs(a), abs(b))
    if g2 > 1:
        a //= g2
        b //= g2
        c //= g2
    return (a, b, c)

def _build_lines(points):
    n = len(points)
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    
    lines = []
    line_masks = []
    point_lines = [[] for _ in range(n)]
    
    for s in line_map.values():
        if len(s) >= 3:
            lines.append(frozenset(s))
            mask = 0
            for idx in s:
                mask |= 1 << idx
                point_lines[idx].append(len(lines) - 1)
            line_masks.append(mask)
            
    return lines, line_masks, point_lines

def _check_no_four(points, lines):
    for L in lines:
        if len(L) >= 4:
            return False
    return True

def _max_cap_free(points, lines, line_masks, point_lines, time_limit=0.5):
    """
    Exact max subset with no 3 collinear using Branch and Bound.
    Optimized for speed.
    """
    n = len(points)
    start = time.time()
    timed_out = [False]
    
    # Order points by degree (number of lines they are on) descending
    # This helps prune branches where high-degree points are excluded
    order = sorted(range(n), key=lambda i: len(point_lines[i]), reverse=True)
    
    best = [0]
    cur = [0]
    line_count = [0] * len(lines)
    
    # Precompute for each point in order, the list of lines it affects
    # and for each line, the points on it (in original indices)
    
    def dfs_order(idx):
        if timed_out[0]:
            return
        if time.time() - start > time_limit:
            timed_out[0] = True
            return
            
        if idx == n:
            if cur[0] > best[0]:
                best[0] = cur[0]
            return
            
        # Bound: even if we add all remaining points, can we beat best?
        if cur[0] + (n - idx) <= best[0]:
            return
            
        p = order[idx]
        
        # Option 1: Skip p
        dfs_order(idx + 1)
        
        if timed_out[0]:
            return
            
        # Option 2: Include p
        can_include = True
        touched = []
        for li in point_lines[p]:
            if line_count[li] >= 2:
                can_include = False
                break
            line_count[li] += 1
            touched.append(li)
            
        if can_include:
            cur[0] += 1
            dfs_order(idx + 1)
            cur[0] -= 1
            for li in touched:
                line_count[li] -= 1
        else:
            for li in touched:
                line_count[li] -= 1

    dfs_order(0)
    return best[0]

def _score(points, time_limit=0.5):
    lines, lm, pl = _build_lines(points)
    if not _check_no_four(points, lines):
        return 999
    return _max_cap_free(points, lines, lm, pl, time_limit)

def _valid(points):
    if len(set(tuple(p) for p in points)) != 20:
        return False
    if any(abs(p[0]) > 100 or abs(p[1]) > 100 for p in points):
        return False
    lines, _, _ = _build_lines(points)
    return _check_no_four(points, lines)

def _generate_initial_points(seed):
    """
    Generate diverse initial points.
    Mix of grid perturbations and random clusters.
    """
    rng = random.Random(seed)
    
    # Strategy 1: Grid perturbation
    if seed % 2 == 0:
        x_coords = [-10, -5, 0, 5, 10]
        y_coords = [-8, -4, 0, 4, 8]
        pts = []
        for x in x_coords:
            for y in y_coords:
                if len(pts) < 20:
                    jx = rng.randint(-2, 2)
                    jy = rng.randint(-2, 2)
                    nx = x + jx
                    ny = y + jy
                    pts.append([nx, ny])
    else:
        # Strategy 2: Random clusters
        pts = []
        used = set()
        while len(pts) < 20:
            cx = rng.randint(-50, 50)
            cy = rng.randint(-50, 50)
            for _ in range(4): # 5 clusters of 4
                if len(pts) >= 20:
                    break
                x = cx + rng.randint(-5, 5)
                y = cy + rng.randint(-5, 5)
                if (x, y) not in used and -100 <= x <= 100 and -100 <= y <= 100:
                    used.add((x, y))
                    pts.append([x, y])
    
    # Ensure distinct and valid
    seen = set()
    final_pts = []
    for p in pts:
        if tuple(p) not in seen:
            seen.add(tuple(p))
            final_pts.append(p)
    
    # If we don't have 20, fill randomly
    while len(final_pts) < 20:
        x = rng.randint(-100, 100)
        y = rng.randint(-100, 100)
        if (x, y) not in seen:
            seen.add((x, y))
            final_pts.append([x, y])
    
    # Fix invalidities (4 collinear)
    for _ in range(100):
        lines, _, _ = _build_lines(final_pts)
        if _check_no_four(final_pts, lines):
            break
        # Move a random point
        idx = rng.randrange(20)
        final_pts[idx] = [rng.randint(-100, 100), rng.randint(-100, 100)]
        
    return final_pts

def _simulated_annealing(points, time_budget, rng):
    """
    Simulated Annealing to minimize the score.
    """
    start = time.time()
    cur = [p[:] for p in points]
    cur_lines, cur_lm, cur_pl = _build_lines(cur)
    if not _check_no_four(cur, cur_lines):
        return cur, 999
    cur_score = _max_cap_free(cur, cur_lines, cur_lm, cur_pl, 0.3)
    
    best = [p[:] for p in cur]
    best_score = cur_score
    
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.99
    
    while time.time() - start < time_budget and T > T_min:
        # Generate a neighbor
        idx = rng.randrange(20)
        old_pt = cur[idx][:]
        
        # Local move
        if rng.random() < 0.7:
            new_pt = [old_pt[0] + rng.randint(-3, 3), old_pt[1] + rng.randint(-3, 3)]
        else:
            new_pt = [rng.randint(-100, 100), rng.randint(-100, 100)]
            
        # Check bounds and collision
        if not (-100 <= new_pt[0] <= 100 and -100 <= new_pt[1] <= 100):
            continue
        collides = False
        for j in range(20):
            if j != idx and cur[j] == new_pt:
                collides = True
                break
        if collides:
            continue
            
        cur[idx] = new_pt
        new_lines, new_lm, new_pl = _build_lines(cur)
        
        if not _check_no_four(cur, new_lines):
            cur[idx] = old_pt
            continue
            
        new_score = _max_cap_free(cur, new_lines, new_lm, new_pl, 0.3)
        
        # Acceptance criterion
        if new_score < best_score:
            best_score = new_score
            best = [p[:] for p in cur]
        elif new_score <= cur_score:
            pass # Always accept improvement or equal
        else:
            delta = new_score - cur_score
            # Accept with probability exp(-delta / T)
            if rng.random() < math.exp(-delta / T):
                pass # Accept
            else:
                cur[idx] = old_pt
                continue
                
        cur_score = new_score
        T *= cooling_rate
        
    return best, best_score

def construct():
    start_time = time.time()
    best_points = None
    best_score = 999
    
    num_starts = 10
    time_per_start = (DEADLINE - 10) / num_starts
    
    for s in range(num_starts):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        pts = _generate_initial_points(s)
        if not _valid(pts):
            continue
            
        rng = random.Random(1000 + s)
        refined, rs = _simulated_annealing(pts, time_per_start, rng)
        
        if rs < best_score:
            best_score = rs
            best_points = refined
            
        if best_score <= 12:
            break
            
    if best_points is None:
        # Fallback
        best_points = [[x, x*x//4] for x in range(-10, 10)]
        
    return best_points
