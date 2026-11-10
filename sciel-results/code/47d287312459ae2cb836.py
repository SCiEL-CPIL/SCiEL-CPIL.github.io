"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
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
            return -1, sz  # invalid
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

def _simulated_annealing(points, start_temp=10.0, end_temp=0.01, max_time=50.0, seed=None):
    """Simulated annealing to maximize 4-point lines."""
    if seed is not None:
        random.seed(seed)
    start_time = time.time()
    current = [list(p) for p in points]
    line_map = _build_line_map(current)
    cur_score, _ = _score_from_line_map(line_map)
    if cur_score < 0:
        cur_score = -100
    
    best = [list(p) for p in current]
    best_score = cur_score if cur_score >= 0 else 0
    if cur_score < 0:
        best_score = 0
        best = [list(p) for p in current]
    
    temp = start_temp
    total_time = max_time
    steps = 0
    
    while time.time() - start_time < max_time:
        temp = max(end_temp, start_temp * (1 - (time.time() - start_time) / total_time))
        
        idx = random.randint(0, 15)
        # Mix of small and medium moves
        r = random.random()
        if r < 0.6:
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
        elif r < 0.9:
            dx = random.randint(-8, 8)
            dy = random.randint(-8, 8)
        else:
            dx = random.randint(-20, 20)
            dy = random.randint(-20, 20)
        
        if dx == 0 and dy == 0:
            continue
        
        new_x = current[idx][0] + dx
        new_y = current[idx][1] + dy
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
        
        new_points = [list(p) for p in current]
        new_points[idx] = [new_x, new_y]
        
        # Check distinctness quickly
        if len(set(tuple(p) for p in new_points)) != 16:
            continue
        
        new_line_map = _build_line_map(new_points)
        new_score, _ = _score_from_line_map(new_line_map)
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
            # Accept with probability exp(delta/temp)
            if temp > 0:
                prob = min(1.0, max(0.0, delta / temp))
                # Use a simple acceptance
                if random.random() < prob:
                    current = new_points
                    cur_score = new_score
        
        steps += 1
    
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
    # Ensure distinctness
    seen = set()
    valid_pts = []
    for p in pts:
        if (p[0], p[1]) not in seen:
            seen.add((p[0], p[1]))
            valid_pts.append(p)
    # If we lost points, add random ones
    while len(valid_pts) < 16:
        nx = random.randint(-5, 8)
        ny = random.randint(-5, 8)
        if (nx, ny) not in seen:
            seen.add((nx, ny))
            valid_pts.append([nx, ny])
    return valid_pts[:16]

def _generate_lattice_structure():
    """Try a different structured starting point."""
    # 4x4 grid shifted
    return [[x, y] for x in range(1, 5) for y in range(1, 5)]

def construct():
    start_time = time.time()
    deadline = 130  # Keep 20s safety margin before 150s limit
    
    random.seed(12345)
    
    # Start with the 4x4 grid
    initial = _generate_grid_4x4()
    best_points = initial
    best_score = _count_4_points(initial)
    if best_score < 0:
        best_score = 0
    
    # Simulated annealing from the grid
    remaining = deadline - (time.time() - start_time)
    if remaining > 5:
        pts, score = _simulated_annealing(initial, start_temp=8.0, end_temp=0.005, 
                                           max_time=min(40, remaining - 5), seed=42)
        if score > best_score:
            best_points = pts
            best_score = score
    
    # Try perturbed grids with shorter annealing
    for trial in range(8):
        if time.time() - start_time > deadline:
            break
        remaining = deadline - (time.time() - start_time)
        if remaining < 10:
            break
        
        perturbed = _generate_perturbed_grid(perturb_prob=0.2 + 0.05 * trial, 
                                             max_perturb=2 + trial // 3)
        # Quick score
        s = _count_4_points(perturbed)
        if s < 0:
            continue
        
        time_per_trial = min(15, remaining / (8 - trial))
        pts, score = _simulated_annealing(perturbed, start_temp=5.0, end_temp=0.01,
                                           max_time=time_per_trial, seed=100 + trial)
        if score > best_score:
            best_points = pts
            best_score = score
    
    # Final local polish: try all small moves on each point
    if time.time() - start_time < deadline - 10:
        current = [list(p) for p in best_points]
        cur_score = _count_4_points(current)
        if cur_score < 0:
            cur_score = 0
        
        improved = True
        while improved and time.time() - start_time < deadline - 10:
            improved = False
            for idx in range(16):
                if time.time() - start_time > deadline - 10:
                    break
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
                        if s > cur_score:
                            current = new_pts
                            cur_score = s
                            if cur_score > best_score:
                                best_points = [list(p) for p in current]
                                best_score = cur_score
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    
    # Final validation
    if not _is_valid(best_points):
        return _generate_grid_4x4()
    return best_points
# EVOLVE-BLOCK-END
