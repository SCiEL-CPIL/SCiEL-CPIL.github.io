"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from math import gcd

def _line_key(p1, p2):
    """Compute canonical line key for two points."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    if g == 0:
        g = 1
    dx_n = dx // g
    dy_n = dy // g
    if dx_n < 0 or (dx_n == 0 and dy_n < 0):
        dx_n = -dx_n
        dy_n = -dy_n
    c = dy_n * p1[0] - dx_n * p1[1]
    return (dx_n, dy_n, c)

def _build_line_map(points):
    """Build dict: line_key -> set of point indices on that line."""
    line_map = {}
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            lk = _line_key(points[i], points[j])
            if lk is None:
                continue
            if lk not in line_map:
                line_map[lk] = set()
            line_map[lk].add(i)
            line_map[lk].add(j)
    return line_map

def _count_triples_from_line_map(line_map):
    """Count total collinear triples from line map."""
    total = 0
    for lk, pts in line_map.items():
        c = len(pts)
        if c >= 3:
            total += c * (c - 1) * (c - 2) // 6
    return total

def _has_four_collinear(line_map):
    """Check if any line has >= 4 points."""
    for lk, pts in line_map.items():
        if len(pts) >= 4:
            return True
    return False

def _largest_no_three_collinear(points):
    """Find the largest subset with no three collinear using branch and bound."""
    n = len(points)
    if n <= 2:
        return n
    
    # Precompute collinearity: for each pair, which other points are on same line
    # Use bitmask approach for speed
    line_map = _build_line_map(points)
    
    # For each point, precompute which pairs among remaining points form a collinear triple with it
    # collinear[i] = set of (j, k) pairs where i, j, k are collinear and j < k
    collinear_with = [set() for _ in range(n)]
    for lk, pts in line_map.items():
        if len(pts) < 3:
            continue
        pts_list = sorted(pts)
        for idx_a in range(len(pts_list)):
            for idx_b in range(idx_a + 1, len(pts_list)):
                for idx_c in range(idx_b + 1, len(pts_list)):
                    a, b, c = pts_list[idx_a], pts_list[idx_b], pts_list[idx_c]
                    collinear_with[a].add((b, c))
                    collinear_with[b].add((a, c))
                    collinear_with[c].add((a, b))
    
    best_size = 0
    
    def branch(subset_mask, candidates):
        nonlocal best_size
        current_size = bin(subset_mask).count('1')
        if current_size > best_size:
            best_size = current_size
        
        if not candidates:
            return
        
        if current_size + len(candidates) <= best_size:
            return
        
        for i, idx in enumerate(candidates):
            # Check if adding idx creates a collinear triple with any two in subset
            valid = True
            for j_bit in range(n):
                if subset_mask & (1 << j_bit):
                    for k_bit in range(j_bit + 1, n):
                        if subset_mask & (1 << k_bit):
                            if (j_bit, k_bit) in collinear_with[idx]:
                                valid = False
                                break
                    if not valid:
                        break
            
            if valid:
                new_mask = subset_mask | (1 << idx)
                new_candidates = candidates[i+1:]
                branch(new_mask, new_candidates)
    
    branch(0, list(range(n)))
    return best_size

def _score_config(points):
    """Score a configuration: must have no 4 collinear, then minimize largest 3-free subset."""
    if len(points) != 20:
        return float('inf')
    line_map = _build_line_map(points)
    if _has_four_collinear(line_map):
        return float('inf')
    return _largest_no_three_collinear(points)

def _make_grid_candidate():
    """Generate a perturbed grid candidate."""
    random.seed(random.randint(0, 2**31))
    pts = []
    seen = set()
    # Try a 4x5 grid with small perturbations
    for i in range(4):
        for j in range(5):
            base_x = (i - 2) * 12
            base_y = (j - 2) * 12
            x = base_x + random.randint(-3, 3)
            y = base_y + random.randint(-3, 3)
            if -100 <= x <= 100 and -100 <= y <= 100 and (x, y) not in seen:
                seen.add((x, y))
                pts.append((x, y))
    while len(pts) < 20:
        x = random.randint(-50, 50)
        y = random.randint(-50, 50)
        if (x, y) not in seen:
            seen.add((x, y))
            pts.append((x, y))
    return pts[:20]

def _make_parabola_candidate():
    """Generate a parabola-based candidate."""
    random.seed(random.randint(0, 2**31))
    a = random.choice([1, -1, 2, -2])
    b = random.randint(-5, 5)
    c = random.randint(-5, 5)
    pts = []
    seen = set()
    x_start = random.randint(-10, 5)
    for x in range(x_start, x_start + 20):
        y = a * x * x + b * x + c
        if -100 <= y <= 100 and (x, y) not in seen:
            seen.add((x, y))
            pts.append((x, y))
    while len(pts) < 20:
        x = random.randint(-50, 50)
        y = random.randint(-50, 50)
        if (x, y) not in seen:
            seen.add((x, y))
            pts.append((x, y))
    return pts[:20]

def _make_random_candidate():
    """Generate a random candidate."""
    pts = []
    seen = set()
    while len(pts) < 20:
        x = random.randint(-50, 50)
        y = random.randint(-50, 50)
        if (x, y) not in seen:
            seen.add((x, y))
            pts.append((x, y))
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin (150s limit, leave 20s buffer)
    
    # Generate initial candidates
    candidates = []
    for _ in range(10):
        candidates.append(_make_grid_candidate())
    for _ in range(10):
        candidates.append(_make_parabola_candidate())
    for _ in range(20):
        candidates.append(_make_random_candidate())
    
    # Filter valid and evaluate
    best_score = float('inf')
    best_points = None
    best_triples = 0
    
    for pts in candidates:
        line_map = _build_line_map(pts)
        if _has_four_collinear(line_map):
            continue
        triples = _count_triples_from_line_map(line_map)
        # Prefer configs with more triples (proxy for smaller 3-free subset)
        if triples > best_triples:
            best_triples = triples
            best_points = pts[:]
    
    if best_points is None:
        # Fallback: find any valid config
        for pts in candidates:
            line_map = _build_line_map(pts)
            if not _has_four_collinear(line_map):
                best_points = pts[:]
                break
        if best_points is None:
            best_points = _make_random_candidate()
    
    # Get initial exact score
    best_score = _score_config(best_points)
    
    # Simulated annealing to maximize collinear triples
    current_points = best_points[:]
    current_triples = best_triples
    current_line_map = _build_line_map(current_points)
    
    T_start = 10.0
    T_end = 0.1
    steps = 50000
    
    for step in range(steps):
        if time.time() > deadline - 5:
            break
        
        # Annealing schedule
        progress = step / steps
        T = T_start * (T_end / T_start) ** progress
        
        # Choose a point to move
        idx = random.randint(0, 19)
        old_pt = current_points[idx]
        
        # Generate new position near old point or random
        if random.random() < 0.7:
            # Local move
            new_x = old_pt[0] + random.randint(-5, 5)
            new_y = old_pt[1] + random.randint(-5, 5)
        else:
            # Global move
            new_x = random.randint(-50, 50)
            new_y = random.randint(-50, 50)
        
        if not (-100 <= new_x <= 100 and -100 <= new_y <= 100):
            continue
        if (new_x, new_y) in current_points:
            continue
        
        # Tentatively move
        current_points[idx] = (new_x, new_y)
        new_line_map = _build_line_map(current_points)
        
        if _has_four_collinear(new_line_map):
            # Revert
            current_points[idx] = old_pt
            continue
        
        new_triples = _count_triples_from_line_map(new_line_map)
        delta = new_triples - current_triples
        
        # Accept if better, or with probability exp(delta/T) if worse
        if delta >= 0:
            accept = True
        else:
            # Avoid overflow: if delta/T is very negative, exp is ~0
            if delta / T < -50:
                accept = False
            else:
                accept = random.random() < math.exp(delta / T)
        
        if accept:
            current_triples = new_triples
            current_line_map = new_line_map
            # Check if this is a new best for exact score
            # Only compute exact score periodically to save time
            if step % 500 == 0 or new_triples > best_triples:
                exact_score = _score_config(current_points)
                if exact_score < best_score:
                    best_score = exact_score
                    best_points = current_points[:]
                    best_triples = new_triples
        else:
            current_points[idx] = old_pt
    
    # Final exact evaluation
    final_score = _score_config(current_points)
    if final_score < best_score:
        best_score = final_score
        best_points = current_points[:]
    
    return [[p[0], p[1]] for p in best_points]
# EVOLVE-BLOCK-END
