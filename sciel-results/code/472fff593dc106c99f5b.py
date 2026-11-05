"""Rewritten program: structural search for 12 integer points minimizing ordinary lines."""

import time
import math
import random
from itertools import combinations

def _count_ordinary_lines(points):
    """Count lines containing exactly two of the given points."""
    n = len(points)
    if n < 3:
        return 0
    line_counts = {}
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            # Normalize the direction to a canonical form
            if dx == 0 and dy == 0:
                continue
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy = -dx, -dy
            elif dx == 0 and dy == 0:
                continue
            g = math.gcd(abs(dx), abs(dy))
            if g != 0:
                dx //= g
                dy //= g
            # Canonical line key: normalized direction + point on line (normalized)
            # A line is uniquely determined by direction (dx,dy) and a point (x0,y0)
            # such that (x0,y0) is the "first" point in some ordering.
            # Better: use the line equation Ax + By + C = 0 with (A,B) normalized.
            A = dy
            B = -dx
            # Normalize (A, B)
            if A == 0 and B == 0:
                continue
            g = math.gcd(abs(A), abs(B))
            if g != 0:
                A //= g
                B //= g
            if A < 0 or (A == 0 and B < 0):
                A, B = -A, -B
            C = -(A * xi + B * yi)
            # Normalize C sign with (A,B) already normalized
            key = (A, B, C)
            line_counts[key] = line_counts.get(key, 0) + 1
    ordinary = 0
    for count in line_counts.values():
        if count == 2:
            ordinary += 1
    return ordinary


def _is_valid(points):
    """Check if points are valid: 12 distinct integer points, not all collinear, |x|,|y| <= 1e9."""
    if len(points) != 12:
        return False
    # Check distinct
    if len(set(tuple(p) for p in points)) != 12:
        return False
    # Check bounds
    for p in points:
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return False
    # Check not all collinear
    if len(points) < 3:
        return False
    p0, p1, p2 = points[0], points[1], points[2]
    cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
    if cross != 0:
        return True  # Not all collinear
    # All three are collinear; check if any point is not on this line
    for p in points[3:]:
        cross = (p1[0] - p0[0]) * (p[1] - p0[1]) - (p1[1] - p0[1]) * (p[0] - p0[0])
        if cross != 0:
            return True
    return False


def _refine(points, max_iters=5000, deadline=None):
    """Local search: try to reduce ordinary lines by moving a single point."""
    if deadline is None:
        deadline = time.time() + 140.0
    points = list(points)
    current_ordinary = _count_ordinary_lines(points)
    best_points = list(points)
    best_ordinary = current_ordinary
    rng = random.Random(42)
    for it in range(max_iters):
        if time.time() > deadline:
            break
        # Pick a random point to move
        idx = rng.randrange(12)
        orig = list(points[idx])
        # Try moving to a random nearby integer coordinate
        for _ in range(20):
            if time.time() > deadline:
                break
            new_x = orig[0] + rng.randint(-50, 50)
            new_y = orig[1] + rng.randint(-50, 50)
            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                continue
            points[idx] = [new_x, new_y]
            if not _is_valid(points):
                points[idx] = orig
                continue
            new_ordinary = _count_ordinary_lines(points)
            if new_ordinary < best_ordinary:
                best_ordinary = new_ordinary
                best_points = list(points)
            points[idx] = orig
    return best_points, best_ordinary


def _try_line_completion(points, deadline=None):
    """Try to place points on lines that already contain 2+ points to reduce ordinary lines."""
    if deadline is None:
        deadline = time.time() + 140.0
    points = list(points)
    current_ordinary = _count_ordinary_lines(points)
    best_points = list(points)
    best_ordinary = current_ordinary
    rng = random.Random(123)
    
    for it in range(2000):
        if time.time() > deadline:
            break
        # Find all lines with exactly 2 points (ordinary lines)
        n = len(points)
        line_map = {}  # line_key -> list of point indices
        for i in range(n):
            xi, yi = points[i]
            for j in range(i + 1, n):
                xj, yj = points[j]
                dx = xj - xi
                dy = yj - yi
                A = dy
                B = -dx
                if A == 0 and B == 0:
                    continue
                g = math.gcd(abs(A), abs(B))
                if g != 0:
                    A //= g
                    B //= g
                if A < 0 or (A == 0 and B < 0):
                    A, B = -A, -B
                C = -(A * xi + B * yi)
                key = (A, B, C)
                if key not in line_map:
                    line_map[key] = []
                line_map[key].append(i)
        
        ordinary_lines = {k: v for k, v in line_map.items() if len(v) == 2}
        if not ordinary_lines:
            break
        
        # Pick a random ordinary line and try to move a third point onto it
        line_key, two_pts = rng.choice(list(ordinary_lines.items()))
        A, B, C = line_key
        # The line equation is A*x + B*y + C = 0
        # Try to move a point not on this line onto it
        candidates = [i for i in range(n) if i not in two_pts]
        if not candidates:
            continue
        idx = rng.choice(candidates)
        orig = list(points[idx])
        # Find an integer point on the line A*x + B*y + C = 0
        # A*x + B*y = -C
        # We can parametrize: if gcd(A,B)=g, solutions exist if g | (-C)
        g = math.gcd(abs(A), abs(B)) if (A != 0 or B != 0) else 1
        if g == 0:
            continue
        if (-C) % g != 0:
            # No integer solution on this line? Shouldn't happen since line has 2 integer points
            continue
        # Find one particular solution
        # Use extended gcd
        def extended_gcd(a, b):
            if b == 0:
                return (a, 1, 0)
            g, x1, y1 = extended_gcd(b, a % b)
            return (g, y1, x1 - (a // b) * y1)
        
        A_abs = abs(A)
        B_abs = abs(B)
        if A_abs == 0:
            # B*y = -C, so y = -C/B
            if (-C) % B_abs == 0:
                y0 = -C // B
                # x is free; pick x near orig[0]
                x0 = orig[0]
                new_pt = [x0, y0]
            else:
                continue
        elif B_abs == 0:
            # A*x = -C
            if (-C) % A_abs == 0:
                x0 = -C // A
                y0 = orig[1]
                new_pt = [x0, y0]
            else:
                continue
        else:
            # A*x + B*y = -C
            g_val, x1, y1 = extended_gcd(A_abs, B_abs)
            if (-C) % g_val != 0:
                continue
            # Particular solution
            x0 = x1 * (-C) // g_val
            y0 = y1 * (-C) // g_val
            # General solution: x = x0 + (B/g)*t, y = y0 - (A/g)*t
            B_g = B_abs // g_val
            A_g = A_abs // g_val
            # Choose t to get close to orig
            t_est = (orig[0] - x0) // B_g if B_g != 0 else 0
            for t in [t_est - 1, t_est, t_est + 1]:
                nx = x0 + B_g * t
                ny = y0 - A_g * t
                if abs(nx) > 10**9 or abs(ny) > 10**9:
                    continue
                new_pt = [nx, ny]
                points[idx] = new_pt
                if not _is_valid(points):
                    points[idx] = orig
                    continue
                # Check new_pt is distinct from all other points
                if any(points[idx] == points[k] for k in range(n) if k != idx):
                    points[idx] = orig
                    continue
                new_ordinary = _count_ordinary_lines(points)
                if new_ordinary < best_ordinary:
                    best_ordinary = new_ordinary
                    best_points = list(points)
                points[idx] = orig
    
    return best_points, best_ordinary


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Start with a 3x4 grid
    points = [[x, y] for x in range(4) for y in range(3)]
    current_ordinary = _count_ordinary_lines(points)
    best_points = list(points)
    best_ordinary = current_ordinary
    
    # Phase 1: Line completion heuristic
    points, ordinary = _try_line_completion(points, deadline=deadline)
    if ordinary < best_ordinary:
        best_ordinary = ordinary
        best_points = list(points)
    
    if time.time() > deadline:
        return best_points
    
    # Phase 2: Local search refinement
    points, ordinary = _refine(points, max_iters=3000, deadline=deadline)
    if ordinary < best_ordinary:
        best_ordinary = ordinary
        best_points = list(points)
    
    if time.time() > deadline:
        return best_points
    
    # Phase 3: Try some structured configurations
    # Configuration: points on a conic or grid-like structure
    structured_configs = [
        # 4x3 grid (already tried)
        [[x, y] for x in range(4) for y in range(3)],
        # 3x4 grid
        [[x, y] for x in range(3) for y in range(4)],
        # Points on two lines with 6 each (but need not all collinear)
        # 6 on x=0, 6 on x=1
        [[0, y] for y in range(6)] + [[1, y] for y in range(6)],
        # Points forming a hexagonal-like pattern
        [[3, 0], [2, 1], [0, 1], [-1, 0], [0, -1], [2, -1],
         [6, 0], [4, 2], [0, 2], [-2, 0], [0, -2], [4, -2]],
        # More compact: use small integer coordinates
        [[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1],[0,2],[1,2],[2,2],[3,2]],
    ]
    
    for config in structured_configs:
        if time.time() > deadline:
            break
        if not _is_valid(config):
            continue
        ord_count = _count_ordinary_lines(config)
        if ord_count < best_ordinary:
            best_ordinary = ord_count
            best_points = list(config)
    
    if time.time() > deadline:
        return best_points
    
    # Phase 4: More extensive local search from best configuration
    points, ordinary = _refine(best_points, max_iters=5000, deadline=deadline)
    if ordinary < best_ordinary:
        best_ordinary = ordinary
        best_points = list(points)
    
    return best_points
