"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
from collections import defaultdict

def _line_key(p1, p2):
    """Return canonical key for the line through p1, p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = math.gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    a = dy
    b = -dx
    c = dx * y1 - dy * x1
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a = -a
        b = -b
        c = -c
    g2 = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
    if g2 > 0:
        a //= g2
        b //= g2
        c //= g2
    return (a, b, c)

def _pairs_from_point(points, idx):
    """Return list of (line_key, other_idx) for all pairs involving point idx."""
    result = []
    for j in range(16):
        if j == idx:
            continue
        key = _line_key(points[idx], points[j])
        if key is not None:
            result.append((key, j))
    return result

def _build_line_map(points):
    """Build line_key -> set of point indices on that line."""
    line_map = defaultdict(set)
    for i in range(16):
        for j in range(i + 1, 16):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_map[key].add(i)
                line_map[key].add(j)
    return line_map

def _score_from_line_map(line_map):
    """Compute (num_4_lines, max_collinear) from line map."""
    num_4 = 0
    max_c = 0
    for key, pts in line_map.items():
        k = len(pts)
        if k == 4:
            num_4 += 1
        if k > max_c:
            max_c = k
    return num_4, max_c

def _score(points):
    """Score: number of 4-point lines, with heavy penalty for 5+ collinear."""
    line_map = _build_line_map(points)
    n4, max_c = _score_from_line_map(line_map)
    if max_c >= 5:
        return n4 - 10000 * (max_c - 4)
    return n4

def _incremental_score(points, idx, line_map):
    """
    After moving point idx, update line_map and return new score.
    Returns (new_score, updated_line_map).
    """
    # Remove old pairs involving idx
    for j in range(16):
        if j == idx:
            continue
        key = _line_key(points[idx], points[j])
        if key is not None and key in line_map:
            line_map[key].discard(idx)
            if not line_map[key]:
                del line_map[key]
    
    # Add new pairs involving idx
    for j in range(16):
        if j == idx:
            continue
        key = _line_key(points[idx], points[j])
        if key is not None:
            line_map[key].add(idx)
            line_map[key].add(j)
    
    n4, max_c = _score_from_line_map(line_map)
    if max_c >= 5:
        return n4 - 10000 * (max_c - 4)
    return n4

def _generate_grid_subset(rows, cols, subset_size=16):
    """Generate a random subset of points from a rows x cols grid."""
    all_points = [[x, y] for x in range(cols) for y in range(rows)]
    if len(all_points) <= subset_size:
        return all_points
    random.shuffle(all_points)
    return all_points[:subset_size]

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Generate diverse initial configurations
    init_configs = []
    
    # 4x4 grid
    init_configs.append([[x, y] for x in range(4) for y in range(4)])
    
    # 5x5 grid subsets (denser structure)
    for _ in range(8):
        init_configs.append(_generate_grid_subset(5, 5, 16))
    
    # 6x4 grid subsets
    for _ in range(5):
        init_configs.append(_generate_grid_subset(6, 4, 16))
    
    # 4x6 grid subsets
    for _ in range(5):
        init_configs.append(_generate_grid_subset(4, 6, 16))
    
    # 7x3 grid subsets
    for _ in range(3):
        init_configs.append(_generate_grid_subset(7, 3, 16))
    
    # Random configurations
    for _ in range(20):
        pts = []
        while len(pts) < 16:
            p = [random.randint(-40, 40), random.randint(-40, 40)]
            if p not in pts:
                pts.append(p)
        init_configs.append(pts)
    
    # Find best initial configuration
    best_score = -1
    best_points = None
    for cfg in init_configs:
        s = _score(cfg)
        if s > best_score:
            best_score = s
            best_points = [p[:] for p in cfg]
    
    # Build line map for best config
    best_line_map = _build_line_map(best_points)
    
    # Simulated annealing with incremental scoring
    current = [p[:] for p in best_points]
    current_score = best_score
    current_line_map = _build_line_map(current)
    
    T = 20.0
    T_min = 0.05
    cooling = 0.9999
    
    iteration = 0
    while time.time() < deadline - 20 and T > T_min:
        # Move: pick a random point and perturb it
        idx = random.randint(0, 15)
        old_x, old_y = current[idx]
        
        # Variable perturbation size based on temperature
        max_pert = max(2, int(8 * T / 20.0))
        new_x = old_x + random.randint(-max_pert, max_pert)
        new_y = old_y + random.randint(-max_pert, max_pert)
        
        # Clamp
        new_x = max(-1000, min(1000, new_x))
        new_y = max(-1000, min(1000, new_y))
        
        # Check if point already exists
        if [new_x, new_y] in current:
            T *= cooling
            continue
        
        old_point = [old_x, old_y]
        current[idx] = [new_x, new_y]
        new_score = _incremental_score(current, idx, current_line_map)
        
        if new_score > current_score or random.random() < math.exp((new_score - current_score) / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current]
                best_line_map = _build_line_map(best_points)
        else:
            current[idx] = old_point
            # Restore line map
            for j in range(16):
                if j == idx:
                    continue
                key = _line_key(current[idx], points_j := current[j])
                if key is not None and key in current_line_map:
                    current_line_map[key].discard(idx)
                    if not current_line_map[key]:
                        del current_line_map[key]
            for j in range(16):
                if j == idx:
                    continue
                key = _line_key(old_point, current[j])
                if key is not None:
                    current_line_map[key].add(idx)
                    current_line_map[key].add(j)
        
        T *= cooling
        iteration += 1
    
    # Local refinement: try all single-point moves with wider range
    # Use incremental scoring for speed
    local_line_map = _build_line_map(best_points)
    for _ in range(20):  # iterations of local search
        improved = False
        for idx in range(16):
            old_x, old_y = best_points[idx]
            for dx in range(-10, 11):
                for dy in range(-10, 11):
                    if dx == 0 and dy == 0:
                        continue
                    nx = old_x + dx
                    ny = old_y + dy
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    if [nx, ny] in best_points:
                        continue
                    old_pt = [old_x, old_y]
                    best_points[idx] = [nx, ny]
                    s = _incremental_score(best_points, idx, local_line_map)
                    if s > best_score:
                        best_score = s
                        improved = True
                    else:
                        best_points[idx] = old_pt
                        # Restore
                        for j in range(16):
                            if j == idx:
                                continue
                            key = _line_key(best_points[idx], best_points[j])
                            if key is not None and key in local_line_map:
                                local_line_map[key].discard(idx)
                                if not local_line_map[key]:
                                    del local_line_map[key]
                        for j in range(16):
                            if j == idx:
                                continue
                            key = _line_key(old_pt, best_points[j])
                            if key is not None:
                                local_line_map[key].add(idx)
                                local_line_map[key].add(j)
                    if time.time() > deadline - 10:
                        return [[int(x), int(y)] for x, y in best_points]
        if not improved:
            break
    
    # Second round of SA from best found so far, with lower temperature
    if time.time() < deadline - 10:
        current = [p[:] for p in best_points]
        current_score = best_score
        current_line_map = _build_line_map(current)
        T = 8.0
        T_min = 0.02
        cooling = 0.99995
        
        while time.time() < deadline - 5 and T > T_min:
            idx = random.randint(0, 15)
            old_x, old_y = current[idx]
            max_pert = max(1, int(5 * T / 8.0))
            new_x = old_x + random.randint(-max_pert, max_pert)
            new_y = old_y + random.randint(-max_pert, max_pert)
            new_x = max(-1000, min(1000, new_x))
            new_y = max(-1000, min(1000, new_y))
            
            if [new_x, new_y] in current:
                T *= cooling
                continue
            
            old_point = [old_x, old_y]
            current[idx] = [new_x, new_y]
            new_score = _incremental_score(current, idx, current_line_map)
            
            if new_score > current_score or random.random() < math.exp((new_score - current_score) / T):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current]
            else:
                current[idx] = old_point
                # Restore line map
                for j in range(16):
                    if j == idx:
                        continue
                    key = _line_key(current[idx], current[j])
                    if key is not None and key in current_line_map:
                        current_line_map[key].discard(idx)
                        if not current_line_map[key]:
                            del current_line_map[key]
                for j in range(16):
                    if j == idx:
                        continue
                    key = _line_key(old_point, current[j])
                    if key is not None:
                        current_line_map[key].add(idx)
                        current_line_map[key].add(j)
            
            T *= cooling
    
    # Final validation
    final_line_map = _build_line_map(best_points)
    n4, max_c = _score_from_line_map(final_line_map)
    if max_c >= 5:
        # Fallback to 4x4 grid
        return [[x, y] for x in range(4) for y in range(4)]
    
    return [[int(x), int(y)] for x, y in best_points]
# EVOLVE-BLOCK-END
