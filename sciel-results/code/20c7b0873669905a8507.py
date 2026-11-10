"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _line_key(p1, p2):
    """Return a normalized key for the line through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return (0, 1, x1)
    if dy == 0:
        return (1, 0, y1)
    if dx < 0:
        dx, dy = -dx, -dy
        x1, y1 = x2, y2
    # Normalize: gcd
    from math import gcd
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    # Line: dy*x - dx*y = dy*x1 - dx*y1
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            line_counts[key] = line_counts.get(key, 0) + 1
    ordinary = 0
    for key, count in line_counts.items():
        if count == 2:
            ordinary += 1
    return ordinary

def _structured_grid():
    """3x4 grid."""
    return [[x, y] for x in range(4) for y in range(3)]

def _structured_hexagon_dual():
    """
    Construct 12 points from a hexagon-like structure.
    Use points that lie on a few lines to minimize ordinary lines.
    """
    # Try: 4 lines, each with 3 points, arranged to maximize coincidences
    # Use a configuration based on the 3x4 grid but perturbed
    # Or use known good configurations
    
    # Configuration: points on 3 parallel lines with 4 points each
    # y=0: (0,0),(1,0),(2,0),(3,0)
    # y=1: (0,1),(1,1),(2,1),(3,1)  
    # y=2: (0,2),(1,2),(2,2),(3,2)
    # This is the 3x4 grid.
    
    # Alternative: use a more symmetric configuration
    # 12 points: 3 on each of 4 lines forming a complete quadrilateral
    
    # Let's try: points at intersections of 4 lines in general position
    # 4 lines give C(4,2)=6 intersection points, need 6 more
    # Not straightforward.
    
    # Try: 2x6 grid
    return [[x, y] for x in range(6) for y in range(2)]

def _structured_pencil():
    """Points on lines through a common point (pencil)."""
    # 12 points on 4 lines through origin, 3 points each
    # Line 1: (1,0), (2,0), (3,0)
    # Line 2: (0,1), (0,2), (0,3)
    # Line 3: (1,1), (2,2), (3,3)
    # Line 4: (1,-1), (2,-2), (3,-3)
    pts = []
    for k in range(1, 4):
        pts.append([k, 0])
        pts.append([0, k])
        pts.append([k, k])
        pts.append([k, -k])
    return pts

def _structured_cubic():
    """Points on a cubic curve - known to have few ordinary lines."""
    # Use points on y = x^2 mod some structure, but need integer points
    # Actually, use a known configuration: the 12 points of the Hessian configuration
    # or use a 4x3 grid with one point moved
    
    # Try 4x3 grid
    return [[x, y] for x in range(3) for y in range(4)]

def _local_search(points, max_time=10.0):
    """Local search: try swapping coordinates or moving points to reduce ordinary lines."""
    start = time.time()
    best_points = [p[:] for p in points]
    best_count = _count_ordinary(best_points)
    
    if best_count <= 6:
        return best_points, best_count
    
    # Local search: try small perturbations
    # For each point, try moving it to nearby integer positions
    # Or try swapping x/y coordinates
    
    improved = True
    while improved and (time.time() - start) < max_time:
        improved = False
        n = len(best_points)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Try swapping point i and j
                old_i = best_points[i][:]
                old_j = best_points[j][:]
                best_points[i] = old_j
                best_points[j] = old_i
                count = _count_ordinary(best_points)
                if count < best_count:
                    best_count = count
                    improved = True
                    if best_count <= 6:
                        return best_points, best_count
                else:
                    best_points[i] = old_i
                    best_points[j] = old_j
        
        # Try moving individual points
        for i in range(n):
            x, y = best_points[i]
            for dx in [-2, -1, 0, 1, 2]:
                for dy in [-2, -1, 0, 1, 2]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if abs(nx) > 10**9 or abs(ny) > 10**9:
                        continue
                    # Check if new position collides
                    new_pt = [nx, ny]
                    if any(new_pt == p for p in best_points):
                        continue
                    old = best_points[i][:]
                    best_points[i] = new_pt
                    count = _count_ordinary(best_points)
                    if count < best_count:
                        best_count = count
                        improved = True
                        if best_count <= 6:
                            return best_points, best_count
                    else:
                        best_points[i] = old
    
    return best_points, best_count

def construct():
    start = time.time()
    deadline = start + 110.0  # Leave 10s safety margin
    
    # Multiple structured starts
    starts = [
        _structured_grid(),
        _structured_hexagon_dual(),
        _structured_pencil(),
        _structured_cubic(),
    ]
    
    best_points = None
    best_count = float('inf')
    
    for s in starts:
        if time.time() > deadline:
            break
        # Ensure distinct points
        seen = set()
        valid_pts = []
        for p in s:
            if tuple(p) not in seen:
                seen.add(tuple(p))
                valid_pts.append(p)
        if len(valid_pts) < 12:
            continue
        
        pts, count = _local_search(valid_pts, max_time=15.0)
        if count < best_count:
            best_count = count
            best_points = pts
            if best_count <= 6:
                break
    
    if best_points is None:
        # Fallback
        best_points = _structured_grid()
    
    return best_points
# EVOLVE-BLOCK-END
