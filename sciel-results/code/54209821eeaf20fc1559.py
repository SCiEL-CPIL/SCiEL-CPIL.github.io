"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def _line_key(p1, p2):
    """Return a normalized key for the line through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A //= g
    B //= g
    C //= g
    # Normalize sign
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    line_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    return sum(1 for pts in line_points.values() if len(pts) == 2)

def _is_valid(points):
    """Check if points are distinct and within bounds."""
    if len(points) != 12:
        return False
    seen = set()
    for x, y in points:
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    # Check not all collinear
    if len(seen) < 3:
        return False
    (x0, y0), (x1, y1), (x2, y2) = list(seen)[:3]
    if (x1 - x0) * (y2 - y0) == (x2 - x0) * (y1 - y0):
        # Check if ALL points are on this line
        for x, y in seen:
            if (x1 - x0) * (y - y0) != (x2 - x0) * (y1 - y0):
                break
        else:
            return False
    return True

def _try_grid_variant(cols, rows, offset_x=0, offset_y=0, scale=1):
    """Generate a grid variant and return (points, score)."""
    points = []
    for x in range(cols):
        for y in range(rows):
            points.append([offset_x + x * scale, offset_y + y * scale])
    if len(points) != 12:
        return None, float('inf')
    if not _is_valid(points):
        return None, float('inf')
    score = _count_ordinary(points)
    return points, score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin (150s limit)
    
    best_points = None
    best_score = float('inf')
    
    # Start with the known optimal 3x4 grid
    base_grid = [[x, y] for x in range(4) for y in range(3)]
    base_score = _count_ordinary(base_grid)
    best_points = [list(p) for p in base_grid]
    best_score = base_score
    
    # Try various structured configurations
    configs_to_try = []
    
    # Different grid orientations and scales
    for cols in range(2, 7):
        for rows in range(2, 7):
            if cols * rows == 12:
                for scale in [1, 2, 3]:
                    for ox in range(-2, 3):
                        for oy in range(-2, 3):
                            configs_to_try.append((cols, rows, ox, oy, scale))
    
    # Try each configuration
    for cols, rows, ox, oy, scale in configs_to_try:
        if time.time() > deadline:
            break
        points, score = _try_grid_variant(cols, rows, ox, oy, scale)
        if points is not None and score < best_score:
            best_score = score
            best_points = [list(p) for p in points]
    
    # Local search: try to improve the best configuration found
    if best_score > 6:  # Only search if we haven't reached the theoretical minimum
        current_points = [list(p) for p in best_points]
        current_score = best_score
        
        max_iterations = 2000
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Pick a random point to move
            idx = random.randint(0, 11)
            cx, cy = current_points[idx]
            
            improved = False
            # Try several random moves in a small neighborhood
            for _ in range(15):
                nx = cx + random.randint(-3, 3)
                ny = cy + random.randint(-3, 3)
                
                # Check validity
                valid = True
                if abs(nx) > 10**9 or abs(ny) > 10**9:
                    valid = False
                else:
                    for j in range(12):
                        if j != idx and current_points[j][0] == nx and current_points[j][1] == ny:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Apply move and evaluate
                current_points[idx][0] = nx
                current_points[idx][1] = ny
                new_score = _count_ordinary(current_points)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_points = [list(p) for p in current_points]
                    break
                else:
                    # Revert
                    current_points[idx][0] = cx
                    current_points[idx][1] = cy
            
            if not improved and random.random() < 0.05:
                # Occasional larger perturbation
                idx = random.randint(0, 11)
                old_x, old_y = current_points[idx][0], current_points[idx][1]
                current_points[idx][0] = random.randint(-10, 10)
                current_points[idx][1] = random.randint(-10, 10)
                # Check validity
                valid = True
                for j in range(12):
                    if j != idx and current_points[j] == current_points[idx]:
                        valid = False
                        break
                if not valid:
                    current_points[idx][0] = old_x
                    current_points[idx][1] = old_y
                    current_score = _count_ordinary(current_points)
    
    # Final validation and cleanup
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(3)]
    
    result = []
    seen = set()
    for p in best_points:
        x, y = int(p[0]), int(p[1])
        if abs(x) > 10**9 or abs(y) > 10**9:
            x, y = 0, 0
        if (x, y) not in seen:
            seen.add((x, y))
            result.append([x, y])
    
    # Pad if needed (shouldn't happen)
    while len(result) < 12:
        x, y = random.randint(-10, 10), random.randint(-10, 10)
        if (x, y) not in seen:
            seen.add((x, y))
            result.append([x, y])
    
    return result[:12]
# EVOLVE-BLOCK-END
