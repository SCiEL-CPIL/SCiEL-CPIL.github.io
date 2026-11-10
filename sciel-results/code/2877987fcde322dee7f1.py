"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
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

def _score_move(points, idx, new_pt):
    """Fast score evaluation for moving point idx to new_pt.
    Only recompute lines involving point idx. Returns (new_score, new_line_map) or (None, None) if invalid."""
    old_pt = points[idx]
    if old_pt == new_pt:
        return None, None
    if abs(new_pt[0]) > 1000 or abs(new_pt[1]) > 1000:
        return None, None
    # Check distinctness
    for i in range(16):
        if i != idx and points[i] == new_pt:
            return None, None
    
    # Build line map for current state
    line_map = _build_line_map(points)
    cur_score, _ = _score_from_line_map(line_map)
    if cur_score < 0:
        return None, None
    
    # Remove all lines involving old point idx
    lines_to_remove = []
    for key, idx_set in list(line_map.items()):
        if idx in idx_set:
            idx_set.discard(idx)
            if len(idx_set) == 0:
                lines_to_remove.append(key)
    for key in lines_to_remove:
        del line_map[key]
    
    # Add new point and recompute lines involving it
    new_points = [list(p) for p in points]
    new_points[idx] = list(new_pt)
    
    for j in range(16):
        if j == idx:
            continue
        key = _line_key(new_pt, new_points[j])
        if key is not None:
            line_map[key].add(idx)
            line_map[key].add(j)
    
    # Score the new state
    new_score, _ = _score_from_line_map(line_map)
    if new_score < 0:
        return None, None
    
    return new_score, line_map

def _simulated_annealing(points, start_temp=10.0, end_temp=0.01, max_time=50.0, seed=None):
    """Simulated annealing to maximize 4-point lines with proper exponential acceptance."""
    if seed is not None:
        random.seed(seed)
    start_time = time.time()
    current = [list(p) for p in points]
    line_map = _build_line_map(current)
    cur_score, _ = _score_from_line_map(line_map)
    if cur_score < 0:
        cur_score = -100
    
    best = [list(p) for p in current]
    best_score = max(0, cur_score)
    
    total_time = max_time
    
    while time.time() - start_time < max_time:
        elapsed = time.time() - start_time
        temp = max(end_temp, start_temp * (1 - elapsed / total_time))
        
        idx = random.randint(0, 15)
        # Mix of small, medium, and large moves
        r = random.random()
        if r < 0.5:
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
        elif r < 0.8:
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
        elif r < 0.95:
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
        
        new_pt = (new_x, new_y)
        new_score, new_line_map = _score_move(current, idx, new_pt)
        if new_score is None:
            continue
        
        delta = new_score - cur_score
        if delta >= 0:
            current[idx] = [new_x, new_y]
            line_map = new_line_map
            cur_score = new_score
            if cur_score > best_score:
                best = [list(p) for p in current]
                best_score = cur_score
        else:
            # Proper exponential acceptance
            if temp > 0.001:
                # Use log to avoid underflow
                log_prob = delta / temp
                if log_prob > -50:  # Avoid underflow
                    if random.random() < math.exp(log_prob):
                        current[idx] = [new_x, new_y]
                        line_map = new_line_map
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
    return [[x, y] for x in range(1, 5) for y in range(1, 5)]

def _generate_random_cluster():
    """Generate a random clustered configuration."""
    pts = []
    seen = set()
    # Create 4 clusters of 4 points
    for c in range(4):
        cx = random.randint(-2, 2)
        cy = random.randint(-2, 2)
        for _ in range(4):
            nx = cx + random.randint(-2, 2)
            ny = cy + random.randint(-2, 2)
            if (nx, ny) not in seen:
                seen.add((nx, ny))
                pts.append([nx, ny])
    # Fill remaining
    while len(pts) < 16:
        nx = random.randint(-5, 5)
        ny = random.randint(-5, 5)
        if (nx, ny) not in seen:
            seen.add((nx, ny))
            pts.append([nx, ny])
    return pts[:16]

def _local_polish(points, max_time=30.0, radius=6):
    """Intensive local search with multiple passes and wider neighborhood."""
    start_time = time.time()
    current = [list(p) for p in points]
    cur_score = _count_4_points(current)
    if cur_score < 0:
        cur_score = 0
    
    best = [list(p) for p in current]
    best_score = cur_score
    
    improved = True
    while improved and time.time() - start_time < max_time:
        improved = False
        for idx in range(16):
            if time.time() - start_time > max_time:
                break
            # Try all moves in radius
            for dx in range(-radius, radius + 1):
                if time.time() - start_time > max_time:
                    break
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx = current[idx][0] + dx
                    ny = current[idx][1] + dy
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    new_pt = (nx, ny)
                    # Quick distinctness check
                    if any(current[i] == new_pt for i in range(16) if i != idx):
                        continue
                    new_pts = [list(p) for p in current]
                    new_pts[idx] = [nx, ny]
                    s = _count_4_points(new_pts)
                    if s < 0:
                        continue
                    if s > cur_score:
                        current = new_pts
                        cur_score = s
                        if cur_score > best_score:
                            best = [list(p) for p in current]
                            best_score = cur_score
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = 130  # Keep 20s safety margin before 150s limit
    
    random.seed(12345)
    
    # Multiple initial configurations
    initial_configs = [
        _generate_grid_4x4(),
        _generate_lattice_structure(),
        _generate_perturbed_grid(perturb_prob=0.1, max_perturb=1),
        _generate_perturbed_grid(perturb_prob=0.3, max_perturb=2),
        _generate_random_cluster(),
    ]
    
    best_points = None
    best_score = -1
    
    # Score all initial configs
    for config in initial_configs:
        if not _is_valid(config):
            continue
        s = _count_4_points(config)
        if s > best_score:
            best_score = s
            best_points = [list(p) for p in config]
    
    if best_points is None:
        best_points = _generate_grid_4x4()
        best_score = 0
    
    # SA from best initial
    remaining = deadline - (time.time() - start_time)
    if remaining > 10:
        pts, score = _simulated_annealing(best_points, start_temp=6.0, end_temp=0.005,
                                           max_time=min(35, remaining - 10), seed=42)
        if score > best_score:
            best_points = pts
            best_score = score
    
    # SA from other good initial configs
    for config in initial_configs:
        if time.time() - start_time > deadline:
            break
        remaining = deadline - (time.time() - start_time)
        if remaining < 15:
            break
        if not _is_valid(config):
            continue
        s = _count_4_points(config)
        if s < best_score - 2:  # Only try if close to best
            continue
        time_per_trial = min(12, remaining / 3)
        pts, score = _simulated_annealing(config, start_temp=5.0, end_temp=0.01,
                                           max_time=time_per_trial, seed=200)
        if score > best_score:
            best_points = pts
            best_score = score
    
    # Final intensive local polish
    if time.time() - start_time < deadline - 5:
        remaining = deadline - (time.time() - start_time) - 5
        pts, score = _local_polish(best_points, max_time=remaining, radius=5)
        if score > best_score:
            best_points = pts
            best_score = score
    
    # Final validation
    if not _is_valid(best_points):
        return _generate_grid_4x4()
    return best_points
# EVOLVE-BLOCK-END
