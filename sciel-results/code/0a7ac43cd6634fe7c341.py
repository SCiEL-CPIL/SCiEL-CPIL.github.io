"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 3:
        return 0
    # Use a set of normalized line representations
    line_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            # Normalize direction
            g = _gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            # Normalize so that dx > 0 or (dx == 0 and dy > 0)
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            # Line is determined by direction and a point on the line
            # Use the point with smaller x (or smaller y if x equal) as anchor
            if x1 < x2 or (x1 == x2 and y1 < y2):
                ax, ay = x1, y1
            else:
                ax, ay = x2, y2
            key = (dx, dy, ax, ay)
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    
    ordinary = 0
    for key, pts in line_points.items():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _valid(points):
    """Check that points are valid: 12 distinct integer points, |x|,|y| <= 10^9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for p in points:
        if len(p) != 2:
            return False
        x, y = p
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    # Check not all collinear
    if len(points) >= 3:
        (x1, y1), (x2, y2), (x3, y3) = points[0], points[1], points[2]
        cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        if cross == 0:
            # Check if all are collinear
            all_collinear = True
            for i in range(3, len(points)):
                x4, y4 = points[i]
                cross_i = (x2 - x1) * (y4 - y1) - (y2 - y1) * (x4 - x1)
                if cross_i != 0:
                    all_collinear = False
                    break
            if all_collinear:
                return False
    return True

def _init_3x4_grid():
    """Initialize with a 3x4 grid pattern."""
    points = []
    for i in range(3):
        for j in range(4):
            points.append([i * 3, j * 3])
    return points

def construct():
    """
    Construct 12 integer points with a minimized number of ordinary lines.
    Strategy: Start from a 3x4 grid and use local search to minimize ordinary lines.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initialize with 3x4 grid
    best_points = _init_3x4_grid()
    best_count = _count_ordinary(best_points)
    
    # Local search
    rng = random.Random(42)
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Try moving a random point to a random new position
        i = rng.randint(0, 11)
        x = rng.randint(-50, 50)
        y = rng.randint(-50, 50)
        
        # Check bounds
        if abs(x) > 10**9 or abs(y) > 10**9:
            continue
        
        # Check duplicate
        if any(p[0] == x and p[1] == y for p in best_points if best_points.index(p) != i):
            continue
        
        # Create new points
        new_points = [p[:] for p in best_points]
        new_points[i] = [x, y]
        
        if not _valid(new_points):
            continue
        
        new_count = _count_ordinary(new_points)
        
        if new_count < best_count:
            best_count = new_count
            best_points = new_points
            no_improve = 0
        else:
            no_improve += 1
    
    # Final polish: try all single-point moves with wider range
    while time.time() < deadline:
        improved = False
        for i in range(12):
            if time.time() >= deadline:
                break
            for dx in range(-30, 31):
                for dy in range(-30, 31):
                    if time.time() >= deadline:
                        break
                    x = best_points[i][0] + dx
                    y = best_points[i][1] + dy
                    if abs(x) > 10**9 or abs(y) > 10**9:
                        continue
                    if any(p[0] == x and p[1] == y for j, p in enumerate(best_points) if j != i):
                        continue
                    
                    new_points = [p[:] for p in best_points]
                    new_points[i] = [x, y]
                    
                    if not _valid(new_points):
                        continue
                    
                    new_count = _count_ordinary(new_points)
                    if new_count < best_count:
                        best_count = new_count
                        best_points = new_points
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_points
# EVOLVE-BLOCK-END
