"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import sys
from itertools import combinations

def _count_ordinary_lines(points):
    """
    Counts the number of lines containing exactly two of the given points.
    Optimized for N=12.
    """
    n = len(points)
    if n < 2:
        return 0
    
    line_points = {}
    
    pts = [tuple(p) for p in points]
    
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            
            dx = x2 - x1
            dy = y2 - y1
            
            g = math.gcd(dx, dy)
            if g == 0:
                continue
            
            dx_n = dx // g
            dy_n = dy // g
            
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
                
            c = dy_n * x1 - dx_n * y1
            
            key = (dx_n, dy_n, c)
            
            if key not in line_points:
                line_points[key] = set()
            
            line_points[key].add(pts[i])
            line_points[key].add(pts[j])
            
    ordinary_count = 0
    for pts_set in line_points.values():
        if len(pts_set) == 2:
            ordinary_count += 1
            
    return ordinary_count

def _get_lines_with_points(points):
    """Returns dict mapping line_key -> set of point indices on that line."""
    n = len(points)
    line_points = {}
    pts = [tuple(p) for p in points]
    
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(dx, dy)
            if g == 0:
                continue
            dx_n = dx // g
            dy_n = dy // g
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            c = dy_n * x1 - dx_n * y1
            key = (dx_n, dy_n, c)
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    
    # For each point, find all lines it's on and their other points
    return line_points

def _try_move_to_line(points, idx, deadline):
    """
    Try to move point idx to lie on a line defined by two other points,
    choosing an integer position that doesn't coincide with existing points.
    Returns list of candidate new positions.
    """
    n = len(points)
    pts = [tuple(p) for p in points]
    candidates = []
    
    for i, j in combinations(range(n), 2):
        if i == idx or j == idx:
            continue
        x1, y1 = pts[i]
        x2, y2 = pts[j]
        dx = x2 - x1
        dy = y2 - y1
        g = math.gcd(dx, dy)
        if g == 0:
            continue
        dx_n = dx // g
        dy_n = dy // g
        
        # Parametric: (x1 + t*dx_n, y1 + t*dy_n) for integer t
        # Need t such that the point is distinct from all existing points
        # and within bounds
        existing = set(pts)
        
        # Try several values of t
        for t in range(-30, 31):
            if t == 0:
                continue  # would be point i
            if t == 1:
                continue  # would be point j (if dx_n,dy_n is unit step)
            nx = x1 + t * dx_n
            ny = y1 + t * dy_n
            if not (-10**9 <= nx <= 10**9 and -10**9 <= ny <= 10**9):
                continue
            if (nx, ny) in existing:
                continue
            candidates.append((nx, ny))
    
    return candidates

def _local_search(points, deadline, max_iters=50000):
    """
    Local search: repeatedly try to move one point to reduce ordinary lines.
    Uses line-based moves: move a point onto a line through two others.
    """
    current = [list(p) for p in points]
    current_score = _count_ordinary_lines(current)
    best = [list(p) for p in current]
    best_score = current_score
    
    n = len(current)
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        # Pick a random point to move
        idx = random.randint(0, n - 1)
        
        # Generate candidates: move to lie on a line through two other points
        candidates = _try_move_to_line(current, idx, deadline)
        
        if not candidates:
            # Fallback: small random perturbation
            ox, oy = current[idx]
            for _ in range(20):
                nx = ox + random.randint(-15, 15)
                ny = oy + random.randint(-15, 15)
                if -10**9 <= nx <= 10**9 and -10**9 <= ny <= 10**9:
                    if all((nx, ny) != tuple(p) for k, p in enumerate(current) if k != idx):
                        candidates.append((nx, ny))
                        break
        
        if not candidates:
            continue
        
        # Pick a random candidate
        nx, ny = random.choice(candidates)
        new_point = [nx, ny]
        
        new_points = [list(p) for p in current]
        new_points[idx] = new_point
        
        new_score = _count_ordinary_lines(new_points)
        
        if new_score < current_score:
            current = new_points
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best = [list(p) for p in current]
                if best_score == 6:
                    break
        elif new_score == current_score and random.random() < 0.3:
            # Accept equal-score moves to escape plateaus
            current = new_points
    
    return best, best_score

def _generate_symmetric_configs():
    """Generate symmetric configurations likely to have many collinearities."""
    configs = []
    
    # 1. 3x4 grid
    configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # 2. 4x3 grid
    configs.append([[x, y] for x in range(3) for y in range(4)])
    
    # 3. 2x6 grid
    configs.append([[x, y] for x in range(6) for y in range(2)])
    
    # 4. 6x2 grid
    configs.append([[x, y] for x in range(2) for y in range(6)])
    
    # 5. Hexagon (6 outer) + 6 inner points forming another hexagon
    # Regular hexagon vertices at distance R, inner at distance r
    for R, r in [(10, 5), (20, 10), (30, 15)]:
        hex_outer = []
        hex_inner = []
        for i in range(6):
            angle = i * math.pi / 3
            hex_outer.append([int(round(R * math.cos(angle))), int(round(R * math.sin(angle)))])
            angle2 = i * math.pi / 3 + math.pi / 6
            hex_inner.append([int(round(r * math.cos(angle2))), int(round(r * math.sin(angle2)))])
        config = hex_outer + hex_inner
        # Check distinct
        seen = set()
        unique = []
        for p in config:
            t = tuple(p)
            if t not in seen:
                seen.add(t)
                unique.append(p)
        if len(unique) == 12:
            configs.append(unique)
    
    # 6. Cube vertices projected (12 points: 8 cube corners + 4 face centers or similar)
    # Actually, use a different approach: 3 parallel lines with 4 points each
    # Line 1: y=0, x=0,1,2,3
    # Line 2: y=1, x=0,1,2,3
    # Line 3: y=2, x=0,1,2,3
    # That's 3x4 grid, already have it.
    
    # 7. 4 lines with 3 points each, arranged in a pattern
    # Lines: y=0 (3 pts), y=1 (3 pts), y=2 (3 pts), y=3 (3 pts) -> 4x3 grid
    
    # 8. Points on a circle with some collinearities
    # Try: 6 points on x-axis, 6 points on y-axis? No, that's too degenerate.
    
    # 9. 12 points as vertices of a regular dodecagon? Probably few collinearities.
    
    # 10. Grid with one point moved to create extra collinearities
    base = [[x, y] for x in range(4) for y in range(3)]
    # Try moving one corner to create a diagonal with 3 points
    for move_idx, new_pos in [(0, [1, 1]), (3, [2, 1]), (8, [1, 2]), (11, [2, 2])]:
        config = [list(p) for p in base]
        config[move_idx] = new_pos
        seen = set()
        unique = []
        for p in config:
            t = tuple(p)
            if t not in seen:
                seen.add(t)
                unique.append(p)
        if len(unique) == 12:
            configs.append(unique)
    
    # 11. Random small integer points with seed
    random.seed(123)
    for seed in range(20):
        random.seed(seed)
        config = []
        seen = set()
        while len(config) < 12:
            x = random.randint(-10, 10)
            y = random.randint(-10, 10)
            t = (x, y)
            if t not in seen:
                seen.add(t)
                config.append([x, y])
        configs.append(config)
    
    return configs

def _is_all_collinear(points):
    """Check if all points lie on a single line."""
    n = len(points)
    if n < 3:
        return True
    # Use first two points to define a line, check all others
    x1, y1 = points[0]
    x2, y2 = points[1]
    dx = x2 - x1
    dy = y2 - y1
    for i in range(2, n):
        x3, y3 = points[i]
        if dx * (y3 - y1) != dy * (x3 - x1):
            return False
    return True

def _count_ordinary_lines_fast(points):
    """
    Fast count of ordinary lines using pair-based line key counting.
    Returns the number of lines containing exactly two points.
    """
    n = len(points)
    if n < 2:
        return 0
    
    line_counts = {}
    pts = [tuple(p) for p in points]
    
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(dx, dy)
            if g == 0:
                continue
            dx_n = dx // g
            dy_n = dy // g
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            c = dy_n * x1 - dx_n * y1
            key = (dx_n, dy_n, c)
            line_counts[key] = line_counts.get(key, 0) + 1
    
    # A line with k points is counted C(k,2) times in line_counts.
    # We need lines with exactly 2 points, which are counted once.
    # But we must be careful: line_counts[key] is the number of pairs on that line.
    # If a line has k points, it contributes C(k,2) to the count for that key.
    # We want to count lines where C(k,2) == 1, i.e., k == 2.
    # However, multiple different lines could theoretically have the same key? No, key uniquely identifies a line.
    # So for each unique line key, the number of points on it is k where C(k,2) = line_counts[key].
    # We need to find k such that k*(k-1)//2 == line_counts[key].
    # Then if k == 2, it's an ordinary line.
    
    ordinary = 0
    for key, pair_count in line_counts.items():
        # Solve k*(k-1)//2 == pair_count for integer k >= 2
        # k = (1 + sqrt(1 + 8*pair_count)) / 2
        disc = 1 + 8 * pair_count
        sqrt_disc = int(math.isqrt(disc))
        if sqrt_disc * sqrt_disc == disc:
            k = (1 + sqrt_disc) // 2
            if k * (k - 1) // 2 == pair_count:
                if k == 2:
                    ordinary += 1
            else:
                # Should not happen, but handle gracefully
                pass
        else:
            # Non-integer k, should not happen for valid point sets
            pass
    
    return ordinary

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    best_points = None
    best_score = float('inf')
    
    # Structural approach: generate configurations from affine grids and known good patterns
    # These are guaranteed to have many collinearities and are not all collinear.
    
    configs = []
    
    # 1. 3x4 grid (12 points)
    configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # 2. 4x3 grid
    configs.append([[x, y] for x in range(3) for y in range(4)])
    
    # 3. 2x6 grid
    configs.append([[x, y] for x in range(6) for y in range(2)])
    
    # 4. 6x2 grid
    configs.append([[x, y] for x in range(2) for y in range(6)])
    
    # 5. 12 points on 3 lines with 4 points each (parallel lines)
    configs.append([[x, 0] for x in range(4)] + [[x, 1] for x in range(4)] + [[x, 2] for x in range(4)])
    
    # 6. 12 points on 4 lines with 3 points each (parallel lines)
    configs.append([[x, 0] for x in range(3)] + [[x, 1] for x in range(3)] + [[x, 2] for x in range(3)] + [[x, 3] for x in range(3)])
    
    # 7. Modified 3x4 grid with one point moved to create extra collinearity
    base = [[x, y] for x in range(4) for y in range(3)]
    for idx, new_pos in [(0, [1, 1]), (3, [2, 1]), (8, [1, 2]), (11, [2, 2]), (1, [1, 2]), (10, [2, 1])]:
        config = [list(p) for p in base]
        config[idx] = new_pos
        seen = set()
        unique = []
        for p in config:
            t = tuple(p)
            if t not in seen:
                seen.add(t)
                unique.append(p)
        if len(unique) == 12:
            configs.append(unique)
    
    # 8. 12 points as vertices of a regular hexagon plus 6 more points creating collinearities
    # Use integer coordinates: hexagon at (0,0), (2,0), (3,1), (2,2), (0,2), (-1,1)
    # Plus 6 more points
    hex_pts = [(0, 0), (2, 0), (3, 1), (2, 2), (0, 2), (-1, 1)]
    # Add points that create collinearities with existing points
    extra_pts = [(1, 0), (1, 2), (3, 0), (3, 2), (0, 1), (-1, 0)]
    config = [list(p) for p in hex_pts] + [list(p) for p in extra_pts]
    seen = set()
    unique = []
    for p in config:
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            unique.append(p)
    if len(unique) == 12:
        configs.append(unique)
    
    # 9. 12 points on a 4x3 grid with diagonal emphasis
    configs.append([[0, 0], [1, 1], [2, 2], [3, 3], [0, 1], [1, 0], [2, 1], [3, 2], [0, 2], [1, 2], [2, 0], [3, 1]])
    
    # 10. 12 points forming a "star" pattern with many 3-point lines
    configs.append([[0, 0], [1, 0], [2, 0], [0, 1], [2, 1], [1, 1], [0, 2], [1, 2], [2, 2], [-1, 0], [3, 0], [1, -1]])
    
    # 11. 12 points on 4 lines through a common point (fan pattern)
    # Line 1: y=0, x=-1,0,1
    # Line 2: y=x, x=-1,0,1
    # Line 3: y=-x, x=-1,0,1
    # Line 4: x=0, y=-1,0,1
    # But (0,0) is shared, so we need 12 distinct points
    # Let's use: 4 lines, each with 3 points, sharing the origin
    # Points: (-1,0), (1,0), (-1,1), (1,1), (-1,-1), (1,-1), (0,-1), (0,1)
    # That's only 8. Add 4 more: (2,0), (-2,0), (0,2), (0,-2)
    configs.append([[-1, 0], [1, 0], [-1, 1], [1, 1], [-1, -1], [1, -1], [0, -1], [0, 1], [2, 0], [-2, 0], [0, 2], [0, -2]])
    
    # 12. 12 points on 6 lines with 2 points each, but arranged to minimize ordinary lines
    # This is hard to construct manually, so use a grid-based approach
    configs.append([[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [2, 1], [0, 2], [1, 2], [2, 2], [3, 0], [3, 1], [3, 2]])
    
    # 13. 12 points on 3 lines with 4 points each, not parallel (forming a triangle)
    # Line 1: y=0, x=0,1,2,3
    # Line 2: x=0, y=0,1,2,3
    # Line 3: x+y=3, points (0,3), (1,2), (2,1), (3,0)
    # But (0,0) and (3,0) and (0,3) are shared
    # Distinct points: (0,0), (1,0), (2,0), (3,0), (0,1), (0,2), (1,2), (2,1), (0,3)
    # That's 9. Add 3 more: (1,1), (2,2), (3,1)
    configs.append([[0, 0], [1, 0], [2, 0], [3, 0], [0, 1], [0, 2], [1, 2], [2, 1], [0, 3], [1, 1], [2, 2], [3, 1]])
    
    # Evaluate all configs
    for config in configs:
        if time.time() > deadline:
            break
        
        # Check distinct points
        seen = set()
        unique = []
        for p in config:
            t = tuple(p)
            if t not in seen:
                seen.add(t)
                unique.append(p)
        if len(unique) != 12:
            continue
        
        # Check bounds
        if any(abs(x) > 10**9 or abs(y) > 10**9 for x, y in unique):
            continue
        
        # Check not all collinear
        if _is_all_collinear(unique):
            continue
        
        score = _count_ordinary_lines_fast(unique)
        if score < best_score:
            best_score = score
            best_points = [list(p) for p in unique]
            if best_score == 6:
                return best_points
    
    # If no valid config found, return a default 3x4 grid
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(3)]
    
    # Final safety check
    if _is_all_collinear(best_points):
        # This should not happen, but handle it
        best_points = [[x, y] for x in range(4) for y in range(3)]
    
    return best_points
# EVOLVE-BLOCK-END
