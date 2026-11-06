"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from math import gcd

def _line_key(p1, p2):
    """Get normalized line key from two points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return (1, 0, x1)  # vertical line x = x1
    if dy == 0:
        return (0, 1, y1)  # horizontal line y = y1
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0:
        dx = -dx
        dy = -dy
    # Line: dy*x - dx*y = dy*x1 - dx*y1
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _count_4_lines(points):
    """Count lines containing exactly 4 of the given points, and check no 5 collinear."""
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            line_counts[key] = line_counts.get(key, 0) + 1
    # Count points per line
    point_line_count = {}
    for key, pair_count in line_counts.items():
        if pair_count >= 6:  # C(4,2)=6 means at least 4 points
            # Count actual points on this line
            dx, dy, c = key
            count = 0
            for p in points:
                x, y = p
                if dy * x - dx * y == c:
                    count += 1
            if count >= 5:
                return -1  # Invalid: 5 collinear
            if count == 4:
                point_line_count[key] = 4
    return len(point_line_count)

def _evaluate(points):
    """Evaluate a configuration: returns (num_4_lines, is_valid)."""
    # Check distinctness
    if len(set(map(tuple, points))) != len(points):
        return (0, False)
    # Check bounds
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return (0, False)
    # Count 4-lines and check no 5-collinear
    n4 = _count_4_lines(points)
    if n4 < 0:
        return (0, False)
    return (n4, True)

def _try_local_search(points, deadline):
    """Try local search: move individual points to improve score."""
    best_points = [p[:] for p in points]
    best_score, _ = _evaluate(best_points)
    if not _ or best_score <= 0:
        return best_points, best_score
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(best_points)):
            orig = best_points[i]
            # Try moving point i to nearby positions
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = orig[0] + dx
                    new_y = orig[1] + dy
                    if abs(new_x) > 1000 or abs(new_y) > 1000:
                        continue
                    candidate = [p[:] for p in best_points]
                    candidate[i] = [new_x, new_y]
                    score, valid = _evaluate(candidate)
                    if valid and score > best_score:
                        best_points = candidate
                        best_score = score
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return best_points, best_score

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with 4x4 grid
    grid = [[x, y] for x in range(4) for y in range(4)]
    best_points, best_score = _try_local_search(grid, deadline)
    
    # Also try centered 4x4 grid
    grid2 = [[x - 1.5, y - 1.5] for x in range(4) for y in range(4)]
    # Convert to integers by scaling
    grid2_int = [[int(x * 2), int(y * 2)] for x, y in grid2]
    if time.time() < deadline:
        p2, s2 = _try_local_search(grid2_int, deadline)
        if s2 > best_score:
            best_points, best_score = p2, s2
    
    # Try a few other structured configurations
    if time.time() < deadline:
        # 5x5 grid minus center and 4 corners? No, that's 21.
        # Try 4x4 with one row shifted
        for shift in range(-3, 4):
            if time.time() >= deadline:
                break
            grid3 = [[x, y] for x in range(4) for y in range(4)]
            # Shift the middle row
            for idx in range(4):
                if grid3[idx][1] == 1:  # second row
                    grid3[idx][0] += shift
            p3, s3 = _try_local_search(grid3, deadline)
            if s3 > best_score:
                best_points, best_score = p3, s3
    
    return best_points
# EVOLVE-BLOCK-END
