"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations
from math import gcd

_start_time = None

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

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
    a = dy
    b = -dx
    c = -(a * x1 + b * y1)
    g2 = gcd(abs(a), abs(b))
    if g2 > 0:
        a //= g2
        b //= g2
        c //= g2
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
        c = -c
    return (a, b, c)

def _get_lines(points):
    lines = {}
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    return lines

def _has_four_collinear(points):
    lines = _get_lines(points)
    for pts in lines.values():
        if len(pts) >= 4:
            return True
    return False

def _count_triples(points):
    lines = _get_lines(points)
    count = 0
    for pts in lines.values():
        n = len(pts)
        if n >= 3:
            count += n * (n - 1) * (n - 2) // 6
    return count

def _max_3free_subset(points):
    n = len(points)
    lines = _get_lines(points)
    triples = []
    for pts in lines.values():
        if len(pts) >= 3:
            idx_list = sorted(pts)
            for combo in combinations(idx_list, 3):
                triples.append(combo)
    
    degree = [0] * n
    for t in triples:
        for i in t:
            degree[i] += 1
    order = sorted(range(n), key=lambda i: -degree[i])
    
    lines_through = [[] for _ in range(n)]
    for key, pts in lines.items():
        if len(pts) >= 3:
            for i in sorted(pts):
                lines_through[i].append(pts)
    
    best_size = [0]
    
    def can_add_fast(p, selected_mask):
        for line_pts in lines_through[p]:
            count_in_line = 0
            for i in line_pts:
                if i != p and (selected_mask >> i) & 1:
                    count_in_line += 1
                    if count_in_line >= 2:
                        return False
        return True
    
    def dfs(idx, selected_mask, current_size):
        if _elapsed() > 115:
            return
        if idx == n:
            if current_size > best_size[0]:
                best_size[0] = current_size
            return
        if current_size + (n - idx) <= best_size[0]:
            return
        point = order[idx]
        dfs(idx + 1, selected_mask, current_size)
        if can_add_fast(point, selected_mask):
            dfs(idx + 1, selected_mask | (1 << point), current_size + 1)
    
    dfs(0, 0, 0)
    return best_size[0]

def _validate(points):
    if len(points) != 20:
        return False
    seen = set()
    for p in points:
        x, y = p
        if abs(x) > 100 or abs(y) > 100:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    if _has_four_collinear(points):
        return False
    return True

def _generate_quadratic_config(seed=42):
    """Generate points from quadratic curves with perturbation to avoid 4-collinear."""
    random.seed(seed)
    points = []
    # Use multiple quadratic curves: y = a*x^2 + b*x + c
    # Each curve contributes ~7 points, total ~20
    # Perturb to ensure no 4 collinear
    
    # Strategy: pick x values and compute y from different quadratics
    # Then perturb slightly to break accidental 4-collinearities
    
    # Use 3 quadratics with 7, 7, 6 points
    x_vals_sets = [
        list(range(-3, 4)),   # -3 to 3 (7 values)
        list(range(-4, 0)),   # -4 to -1 (4 values) + 3 more
        list(range(1, 7)),    # 1 to 6 (6 values)
    ]
    
    # Actually, let's use a simpler approach:
    # Pick 20 x-values spread out, assign each to a quadratic
    # y = a_i * x^2 + b_i * x + c_i where (a_i, b_i, c_i) varies per point
    
    # Better: use the fact that 3 points determine a parabola
    # So any 3 of our points define a unique parabola
    # If we ensure no 4 points lie on the same parabola, we're good
    # But we WANT many 3-collinear (on a line, not parabola)
    
    # Actually, for collinearity on LINES:
    # Points on y = ax^2 + bx + c are NOT collinear (they're on a parabola)
    # So this approach doesn't help for line collinearity!
    
    # Let me reconsider. The key is to create many LINES with exactly 3 points.
    # One approach: use a near-grid with perturbations.
    
    # Let's use a structured approach:
    # Start with a 4x5 grid, then perturb each point slightly
    # The grid has many 3-collinear (rows, columns, diagonals)
    # Perturbation breaks 4-collinear while preserving most 3-collinear
    
    base_points = []
    for r in range(4):
        for c in range(5):
            x = c * 20 - 40  # -40, -20, 0, 20, 40
            y = r * 25 - 37  # -37, -12, 13, 38
            base_points.append([x, y])
    
    # Perturb each point by a small random amount
    for i in range(20):
        base_points[i][0] += random.randint(-2, 2)
        base_points[i][1] += random.randint(-2, 2)
    
    # Check for 4-collinear and adjust
    for attempt in range(100):
        if not _has_four_collinear(base_points):
            break
        # Find a point that's part of a 4-collinear line and move it
        lines = _get_lines(base_points)
        for key, pts in lines.items():
            if len(pts) >= 4:
                # Move one of the points
                idx = random.choice(sorted(pts))
                base_points[idx][0] += random.randint(-5, 5)
                base_points[idx][1] += random.randint(-5, 5)
                # Keep in bounds
                base_points[idx][0] = max(-100, min(100, base_points[idx][0]))
                base_points[idx][1] = max(-100, min(100, base_points[idx][1]))
                break
    
    return base_points

def _local_search_optimize(points, time_budget=100):
    """Local search to minimize the max 3-free subset size."""
    deadline = time.time() + time_budget
    current = [list(p) for p in points]
    
    # Compute initial score
    current_score = _max_3free_subset(current)
    best = [list(p) for p in current]
    best_score = current_score
    
    random.seed(12345)
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 19)
        old_x, old_y = current[idx]
        
        # Try several candidate moves
        found_improvement = False
        for _ in range(40):
            # Random new position near the old one
            new_x = old_x + random.randint(-8, 8)
            new_y = old_y + random.randint(-8, 8)
            if abs(new_x) > 100 or abs(new_y) > 100:
                continue
            
            # Check distinctness
            valid = True
            for j in range(20):
                if j != idx and current[j][0] == new_x and current[j][1] == new_y:
                    valid = False
                    break
            if not valid:
                continue
            
            # Apply move
            current[idx] = [new_x, new_y]
            
            # Check 4-collinear
            if _has_four_collinear(current):
                current[idx] = [old_x, old_y]
                continue
            
            # Evaluate new score
            new_score = _max_3free_subset(current)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best = [list(p) for p in current]
                found_improvement = True
                break
            else:
                current[idx] = [old_x, old_y]
        
        if not found_improvement:
            # Try a larger perturbation
            for _ in range(15):
                new_x = old_x + random.randint(-15, 15)
                new_y = old_y + random.randint(-15, 15)
                if abs(new_x) > 100 or abs(new_y) > 100:
                    continue
                valid = True
                for j in range(20):
                    if j != idx and current[j][0] == new_x and current[j][1] == new_y:
                        valid = False
                        break
                if not valid:
                    continue
                current[idx] = [new_x, new_y]
                if _has_four_collinear(current):
                    current[idx] = [old_x, old_y]
                    continue
                new_score = _max_3free_subset(current)
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best = [list(p) for p in current]
                    break
                else:
                    current[idx] = [old_x, old_y]
    
    return best, best_score

def _multi_start_search(time_budget=90):
    """Try multiple initial configurations and refine each."""
    deadline = time.time() + time_budget
    best_points = None
    best_score = float('inf')
    
    # Generate diverse initial configurations
    configs = []
    
    # Config 1: Perturbed 4x5 grid
    configs.append(_generate_quadratic_config(42))
    
    # Config 2: Different perturbation seed
    configs.append(_generate_quadratic_config(123))
    
    # Config 3: 5x4 grid orientation
    random.seed(789)
    base = []
    for r in range(5):
        for c in range(4):
            x = c * 25 - 37
            y = r * 20 - 40
            base.append([x + random.randint(-2, 2), y + random.randint(-2, 2)])
    for attempt in range(50):
        if not _has_four_collinear(base):
            break
        lines = _get_lines(base)
        for key, pts in lines.items():
            if len(pts) >= 4:
                idx = random.choice(sorted(pts))
                base[idx][0] += random.randint(-5, 5)
                base[idx][1] += random.randint(-5, 5)
                base[idx][0] = max(-100, min(100, base[idx][0]))
                base[idx][1] = max(-100, min(100, base[idx][1]))
                break
    configs.append(base)
    
    # Config 4: Random dense cluster
    random.seed(321)
    pts = []
    for i in range(20):
        x = random.randint(-30, 30)
        y = random.randint(-30, 30)
        pts.append([x, y])
    for attempt in range(200):
        if not _has_four_collinear(pts):
            break
        lines = _get_lines(pts)
        for key, lp in lines.items():
            if len(lp) >= 4:
                idx = random.choice(sorted(lp))
                pts[idx][0] += random.randint(-8, 8)
                pts[idx][1] += random.randint(-8, 8)
                pts[idx][0] = max(-100, min(100, pts[idx][0]))
                pts[idx][1] = max(-100, min(100, pts[idx][1]))
                break
    configs.append(pts)
    
    # Refine each configuration
    for i, config in enumerate(configs):
        if time.time() > deadline - 15:
            break
        if not _validate(config):
            continue
        
        remaining = deadline - time.time() - 5
        if remaining < 10:
            break
        
        refined, score = _local_search_optimize(config, time_budget=min(remaining, 30))
        
        if _validate(refined):
            if score < best_score:
                best_score = score
                best_points = refined
    
    return best_points, best_score

def construct():
    global _start_time
    _start_time = time.time()
    random.seed(42)
    
    # Multi-start local search
    points, score = _multi_start_search(time_budget=105)
    
    # Fallback if multi-start fails
    if points is None:
        points = _generate_quadratic_config(42)
        if not _validate(points):
            return [[x, x * x] for x in range(-10, 10)]
    
    # Final validation
    if not _validate(points):
        return [[x, x * x] for x in range(-10, 10)]
    
    return points
# EVOLVE-BLOCK-END
