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
        elapsed = time.time() - start_time
        temp = max(end_temp, start_temp * (1 - elapsed / total_time))
        
        move_type = random.random()
        
        if move_type < 0.7:
            # Single point move
            idx = random.randint(0, 15)
            r = random.random()
            if r < 0.5:
                dx = random.randint(-4, 4)
                dy = random.randint(-4, 4)
            elif r < 0.85:
                dx = random.randint(-10, 10)
                dy = random.randint(-10, 10)
            else:
                dx = random.randint(-25, 25)
                dy = random.randint(-25, 25)
            
            if dx == 0 and dy == 0:
                continue
            
            new_x = current[idx][0] + dx
            new_y = current[idx][1] + dy
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
            
            new_points = [list(p) for p in current]
            new_points[idx] = [new_x, new_y]
        else:
            # Swap two points
            idx1, idx2 = random.sample(range(16), 2)
            new_points = [list(p) for p in current]
            new_points[idx1], new_points[idx2] = current[idx2][:], current[idx1][:]
        
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
            if temp > 0.001:
                import math
                prob = math.exp(delta / temp)
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

def _generate_hexagonal():
    """Generate a hexagonal-like arrangement."""
    pts = []
    # 4x4 grid with hexagonal offset
    for row in range(4):
        for col in range(4):
            x = col
            y = row
            if row % 2 == 1:
                x = col + 1
            pts.append([x, y])
    return pts

def _generate_two_grids():
    """Two 3x3 grids with one shared point, or 4x4 with perturbation."""
    # 4x4 grid with one point moved
    pts = [[x, y] for x in range(4) for y in range(4)]
    # Move one corner point
    pts[0] = [4, 1]  # Move (0,0) to (4,1)
    return pts

def _generate_lattice_structure():
    """Try a different structured starting point."""
    # 4x4 grid shifted
    return [[x, y] for x in range(1, 5) for y in range(1, 5)]

def construct():
    start_time = time.time()
    deadline = 140  # Keep 10s safety margin before 150s limit
    
    random.seed(12345)
    
    # Try multiple initial configurations
    initial_configs = [
        ("grid4x4", _generate_grid_4x4()),
        ("hex", _generate_hexagonal()),
        ("two_grids", _generate_two_grids()),
        ("perturb1", _generate_perturbed_grid(perturb_prob=0.3, max_perturb=2)),
        ("perturb2", _generate_perturbed_grid(perturb_prob=0.5, max_perturb=3)),
        ("perturb3", _generate_perturbed_grid(perturb_prob=0.2, max_perturb=1)),
    ]
    
    best_points = None
    best_score = -1
    
    # Score all initial configs
    for name, config in initial_configs:
        s = _count_4_points(config)
        if s > best_score:
            best_score = s
            best_points = [list(p) for p in config]
    
    if best_points is None or best_score < 0:
        best_points = _generate_grid_4x4()
        best_score = _count_4_points(best_points)
        if best_score < 0:
            best_score = 0
    
    # Simulated annealing from the best initial config
    remaining = deadline - (time.time() - start_time)
    if remaining > 10:
        pts, score = _simulated_annealing(best_points, start_temp=10.0, end_temp=0.005, 
                                           max_time=min(50, remaining - 10), seed=42)
        if score > best_score:
            best_points = pts
            best_score = score
    
    # Try annealing from other promising configs
    for name, config in initial_configs:
        if time.time() - start_time > deadline - 15:
            break
        remaining = deadline - (time.time() - start_time)
        if remaining < 15:
            break
        
        s = _count_4_points(config)
        if s < 0:
            continue
        
        time_per_trial = min(20, remaining / 3)
        pts, score = _simulated_annealing(config, start_temp=8.0, end_temp=0.01,
                                           max_time=time_per_trial, seed=200 + hash(name) % 100)
        if score > best_score:
            best_points = pts
            best_score = score
    
    # Final local polish: try all small moves on each point
    if time.time() - start_time < deadline - 15:
        current = [list(p) for p in best_points]
        cur_score = _count_4_points(current)
        if cur_score < 0:
            cur_score = 0
        
        improved = True
        while improved and time.time() - start_time < deadline - 15:
            improved = False
            for idx in range(16):
                if time.time() - start_time > deadline - 15:
                    break
                for dx in range(-8, 9):
                    for dy in range(-8, 9):
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
