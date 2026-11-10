import time
import random
from collections import defaultdict
from math import gcd

def _line_key(p1, p2):
    """Return a canonical key for the line through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    if dx == 0:
        return (0, 0, x1)
    if dy == 0:
        return (0, 1, y1)
    a = dy
    b = -dx
    c = a * x1 + b * y1
    g = gcd(abs(a), abs(b))
    if g > 1:
        a //= g
        b //= g
        c //= g
    if a < 0:
        a = -a
        b = -b
        c = -c
    elif a == 0 and b < 0:
        b = -b
        c = -c
    return (a, b, c)

def _build_line_map(points):
    """Build line map: line_key -> set of point indices."""
    n = len(points)
    line_map = defaultdict(set)
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            key = _line_key(pi, pj)
            if key is not None:
                line_map[key].add(i)
                line_map[key].add(j)
    return line_map

def _score_from_line_map(line_map):
    """Count lines with exactly 4 points, and check no 5+ collinear."""
    count_4 = 0
    max_collinear = 0
    for key, idx_set in line_map.items():
        sz = len(idx_set)
        if sz >= 5:
            return -1, sz
        if sz == 4:
            count_4 += 1
        if sz > max_collinear:
            max_collinear = sz
    return count_4, max_collinear

def _count_4_points(points):
    """Count lines containing exactly 4 of the given points. Returns -1 if invalid."""
    line_map = _build_line_map(points)
    count_4, _ = _score_from_line_map(line_map)
    return count_4

def _is_valid(points):
    """Check validity: 16 distinct, bounds, no 5 collinear."""
    if len(points) != 16:
        return False
    seen = set()
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    line_map = _build_line_map(points)
    for key, idx_set in line_map.items():
        if len(idx_set) >= 5:
            return False
    return True

def _local_search(points, max_iters=1000):
    """Greedy local search: move points to improve score."""
    current = [list(p) for p in points]
    cur_score = _count_4_points(current)
    if cur_score < 0:
        return current, 0
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for idx in range(16):
            best_new_score = cur_score
            best_new_point = current[idx]
            # Try small moves
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    nx = current[idx][0] + dx
                    ny = current[idx][1] + dy
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    new_pts = [list(p) for p in current]
                    new_pts[idx] = [nx, ny]
                    if len(set(tuple(p) for p in new_pts)) != 16:
                        continue
                    s = _count_4_points(new_pts)
                    if s < 0:
                        continue
                    if s > best_new_score:
                        best_new_score = s
                        best_new_point = [nx, ny]
            if best_new_point != current[idx]:
                current[idx] = best_new_point
                cur_score = best_new_score
                improved = True
    return current, cur_score

def _simulated_annealing(points, start_temp=10.0, end_temp=0.01, max_time=10.0, seed=None):
    """Simulated annealing to maximize 4-point lines."""
    if seed is not None:
        random.seed(seed)
    start_time = time.time()
    current = [list(p) for p in points]
    cur_score = _count_4_points(current)
    if cur_score < 0:
        cur_score = -100
    
    best = [list(p) for p in current]
    best_score = cur_score if cur_score >= 0 else 0
    
    while time.time() - start_time < max_time:
        elapsed = time.time() - start_time
        temp = max(end_temp, start_temp * (1 - elapsed / max_time))
        
        idx = random.randint(0, 15)
        r = random.random()
        if r < 0.7:
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
        else:
            dx = random.randint(-10, 10)
            dy = random.randint(-10, 10)
        
        if dx == 0 and dy == 0:
            continue
        
        new_x = current[idx][0] + dx
        new_y = current[idx][1] + dy
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
        
        new_points = [list(p) for p in current]
        new_points[idx] = [new_x, new_y]
        
        if len(set(tuple(p) for p in new_points)) != 16:
            continue
        
        new_score = _count_4_points(new_points)
        if new_score < 0:
            new_score = -100
        
        delta = new_score - cur_score
        if delta >= 0:
            current = new_points
            cur_score = new_score
            if cur_score > best_score:
                best = [list(p) for p in current]
                best_score = cur_score
        else:
            if temp > 0:
                # Use Boltzmann distribution approximation for speed
                prob = min(1.0, max(0.0, delta / temp))
                if random.random() < prob:
                    current = new_points
                    cur_score = new_score
    
    return best, best_score

def _generate_grid_4x4():
    return [[x, y] for x in range(4) for y in range(4)]

def _generate_perturbed_grid(perturb_prob=0.3, max_perturb=3):
    """Generate a perturbed 4x4 grid."""
    pts = []
    for x in range(4):
        for y in range(4):
            if random.random() < perturb_prob:
                nx = x + random.randint(-max_perturb, max_perturb)
                ny = y + random.randint(-max_perturb, max_perturb)
            else:
                nx, ny = x, y
            pts.append([nx, ny])
    seen = set()
    valid_pts = []
    for p in pts:
        if (p[0], p[1]) not in seen:
            seen.add((p[0], p[1]))
            valid_pts.append(p)
    while len(valid_pts) < 16:
        nx = random.randint(-5, 8)
        ny = random.randint(-5, 8)
        if (nx, ny) not in seen:
            seen.add((nx, ny))
            valid_pts.append([nx, ny])
    return valid_pts[:16]

def construct():
    start_time = time.time()
    deadline = 140  # Keep 10s safety margin before 150s limit
    
    random.seed(12345)
    
    best_points = _generate_grid_4x4()
    best_score = _count_4_points(best_points)
    if best_score < 0:
        best_score = 0
    
    # 1. Local search from grid
    pts, score = _local_search(best_points, max_iters=200)
    if score > best_score:
        best_points = pts
        best_score = score
    
    # 2. Simulated annealing from grid
    remaining = deadline - (time.time() - start_time)
    if remaining > 5:
        pts, score = _simulated_annealing(best_points, start_temp=8.0, end_temp=0.005, 
                                           max_time=min(20, remaining - 5), seed=42)
        if score > best_score:
            best_points = pts
            best_score = score
    
    # 3. Try perturbed grids
    for trial in range(10):
        if time.time() - start_time > deadline:
            break
        remaining = deadline - (time.time() - start_time)
        if remaining < 5:
            break
        
        perturbed = _generate_perturbed_grid(perturb_prob=0.2 + 0.05 * trial, 
                                             max_perturb=2 + trial // 2)
        
        # Quick local search
        pts, score = _local_search(perturbed, max_iters=50)
        if score > best_score:
            best_points = pts
            best_score = score
        
        # Short SA
        time_per_trial = min(5, remaining / (10 - trial))
        if time_per_trial > 1:
            pts, score = _simulated_annealing(pts, start_temp=5.0, end_temp=0.01,
                                               max_time=time_per_trial, seed=100 + trial)
            if score > best_score:
                best_points = pts
                best_score = score
    
    # 4. Final polish
    if time.time() - start_time < deadline - 5:
        pts, score = _local_search(best_points, max_iters=500)
        if score > best_score:
            best_points = pts
            best_score = score
    
    # Final validation
    if not _is_valid(best_points):
        return _generate_grid_4x4()
    return best_points
