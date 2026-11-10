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

def _precompute_grid_lines(grid_size, offset=0):
    """Precompute all lines with >= 4 points in a grid_size x grid_size grid.
    Returns:
        all_points: list of (x, y) tuples
        lines_4plus: dict mapping frozenset of point indices -> count
        five_plus_lines: set of frozensets with >= 5 points
    """
    all_points = [(x + offset, y + offset) for x in range(grid_size) for y in range(grid_size)]
    n = len(all_points)
    
    # Build line map for the full grid
    line_map = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(all_points[i], all_points[j])
            if key is not None:
                line_map[key].add(i)
                line_map[key].add(j)
    
    lines_4plus = {}
    five_plus_lines = set()
    for key, idx_set in line_map.items():
        sz = len(idx_set)
        if sz >= 4:
            lines_4plus[frozenset(idx_set)] = sz
            if sz >= 5:
                five_plus_lines.add(frozenset(idx_set))
    
    return all_points, lines_4plus, five_plus_lines

def _subset_score(selected_indices, lines_4plus, five_plus_lines):
    """Score a subset of indices: count 4-point lines fully contained, check validity."""
    sel = frozenset(selected_indices)
    
    # Check validity: no 5-collinear
    for five_line in five_plus_lines:
        if five_line.issubset(sel):
            return -1
    
    # Count 4-point lines
    count_4 = 0
    for line_pts, line_sz in lines_4plus.items():
        if line_sz == 4:
            if line_pts.issubset(sel):
                count_4 += 1
        elif line_sz > 4:
            # If we have exactly 4 of these points, it counts
            intersection = len(line_pts & sel)
            if intersection == 4:
                count_4 += 1
    
    return count_4

def _generate_4x4_grid():
    return [[x, y] for x in range(4) for y in range(4)]

def _generate_perturbed_grid(perturb_prob=0.3, max_perturb=3, seed=None):
    """Generate a perturbed 4x4 grid."""
    if seed is not None:
        random.seed(seed)
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

def _simulated_annealing_subset(all_points, lines_4plus, five_plus_lines, 
                                  initial_indices, start_temp=5.0, end_temp=0.01, 
                                  max_time=30.0, seed=None):
    """Simulated annealing on subset selection from a bounded grid."""
    if seed is not None:
        random.seed(seed)
    start_time = time.time()
    
    n_grid = len(all_points)
    current = list(initial_indices)
    cur_score = _subset_score(current, lines_4plus, five_plus_lines)
    if cur_score < 0:
        cur_score = -100
    
    best = list(current)
    best_score = max(0, cur_score)
    
    temp = start_temp
    total_time = max_time
    
    while time.time() - start_time < max_time:
        elapsed = time.time() - start_time
        temp = max(end_temp, start_temp * (1 - elapsed / total_time))
        
        # Pick a random point in current and a random point not in current
        in_idx = random.choice(current)
        out_candidates = [i for i in range(n_grid) if i not in current]
        if not out_candidates:
            break
        out_idx = random.choice(out_candidates)
        
        new_current = list(current)
        new_current.remove(in_idx)
        new_current.append(out_idx)
        
        new_score = _subset_score(new_current, lines_4plus, five_plus_lines)
        if new_score < 0:
            new_score = -100
        
        delta = new_score - cur_score
        if delta >= 0:
            current = new_current
            cur_score = new_score
            if cur_score > best_score:
                best = list(current)
                best_score = cur_score
        else:
            if temp > 0.001:
                prob = math.exp(delta / temp)
                if random.random() < prob:
                    current = new_current
                    cur_score = new_score
    
    return best, best_score

def _free_space_annealing(points, start_temp=10.0, end_temp=0.005, max_time=30.0, seed=None):
    """Simulated annealing in free space (move individual points)."""
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
    
    temp = start_temp
    total_time = max_time
    
    while time.time() - start_time < max_time:
        elapsed = time.time() - start_time
        temp = max(end_temp, start_temp * (1 - elapsed / total_time))
        
        move_type = random.random()
        
        if move_type < 0.7:
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
            idx1, idx2 = random.sample(range(16), 2)
            new_points = [list(p) for p in current]
            new_points[idx1], new_points[idx2] = current[idx2][:], current[idx1][:]
        
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
            if temp > 0.001:
                prob = math.exp(delta / temp)
                if random.random() < prob:
                    current = new_points
                    cur_score = new_score
    
    return best, best_score

def _local_polish(points, deadline):
    """Final local polish: try all small moves on each point."""
    current = [list(p) for p in points]
    cur_score = _count_4_points(current)
    if cur_score < 0:
        cur_score = 0
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(16):
            if time.time() > deadline:
                break
            for dx in range(-6, 7):
                if time.time() > deadline:
                    break
                for dy in range(-6, 7):
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
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return current, cur_score

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s limit
    
    random.seed(12345)
    
    best_points = None
    best_score = -1
    
    # Strategy 1: Grid-based subset selection (inspiration 1)
    # Try different grid sizes
    grid_configs = [
        (5, 0),   # 5x5 grid at origin
        (6, 0),   # 6x6 grid at origin
        (7, 0),   # 7x7 grid at origin
        (7, -1),  # 7x7 grid shifted
        (8, 0),   # 8x8 grid at origin
    ]
    
    for grid_size, offset in grid_configs:
        if time.time() > deadline - 20:
            break
        
        all_points, lines_4plus, five_plus_lines = _precompute_grid_lines(grid_size, offset)
        n_grid = len(all_points)
        
        if n_grid < 16:
            continue
        
        # Generate multiple initial subsets
        initial_subsets = []
        
        # Start with a random 16-subset
        for trial in range(3):
            random.seed(1000 + trial * 7)
            init = random.sample(range(n_grid), 16)
            initial_subsets.append(init)
        
        # Also try the 4x4 grid pattern mapped into the grid
        if grid_size >= 4:
            grid4x4_indices = [(x * (n_grid // 4) + y) for x in range(4) for y in range(4)]
            # Map to actual grid indices: point (gx, gy) has index gy * grid_size + gx
            grid4x4_indices = [(x * grid_size + y) for x in range(4) for y in range(4)]
            if len(set(grid4x4_indices)) == 16:
                initial_subsets.append(grid4x4_indices)
        
        for init in initial_subsets:
            if time.time() > deadline - 15:
                break
            
            remaining = deadline - time.time()
            time_per_trial = min(25, remaining / 2)
            
            indices, score = _simulated_annealing_subset(
                all_points, lines_4plus, five_plus_lines,
                init, start_temp=5.0, end_temp=0.01,
                max_time=time_per_trial, seed=random.randint(0, 10000)
            )
            
            if score > best_score:
                best_score = score
                best_points = [[all_points[i][0], all_points[i][1]] for i in indices]
    
    # Strategy 2: Free-space simulated annealing from best found
    if best_points is None:
        best_points = _generate_4x4_grid()
        best_score = _count_4_points(best_points)
        if best_score < 0:
            best_score = 0
    
    if time.time() < deadline - 20:
        remaining = deadline - time.time()
        time_for_anneal = min(35, remaining - 20)
        if time_for_anneal > 5:
            pts, score = _free_space_annealing(best_points, start_temp=10.0, end_temp=0.005,
                                                max_time=time_for_anneal, seed=42)
            if score > best_score:
                best_points = pts
                best_score = score
    
    # Also try from the plain 4x4 grid if we haven't already
    grid4x4 = _generate_4x4_grid()
    grid4x4_score = _count_4_points(grid4x4)
    if grid4x4_score > best_score:
        best_points = grid4x4
        best_score = grid4x4_score
    
    # Final local polish
    if time.time() < deadline - 5:
        best_points, best_score = _local_polish(best_points, deadline - 2)
    
    # Final validation
    if not _is_valid(best_points):
        return _generate_4x4_grid()
    
    return best_points
# EVOLVE-BLOCK-END
