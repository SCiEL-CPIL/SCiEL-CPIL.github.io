import time
import sys
import math
from math import gcd
from collections import defaultdict
import random

def _count_ordinary_lines(points):
    """Count the number of lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 3:
        return 0
    
    # Map each line (as a normalized tuple) to the set of point indices on it
    line_map = defaultdict(set)
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            # Normalize the direction
            g = gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            a = dy
            b = -dx
            c = a * x1 + b * y1
            # Normalize (a, b, c) so that gcd(a,b,c) = 1 and a > 0 or (a == 0 and b > 0)
            g2 = gcd(abs(a), abs(b))
            if g2 > 0:
                a //= g2
                b //= g2
                c //= g2
            # Ensure canonical sign
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
                c = -c
            line_map[(a, b, c)].add(i)
            line_map[(a, b, c)].add(j)
    
    # Count lines with exactly 2 points
    ordinary = 0
    for line, pts in line_map.items():
        if len(pts) == 2:
            ordinary += 1
    
    return ordinary

def _try_config(points):
    """Validate and count ordinary lines for a configuration."""
    if len(points) != 12:
        return None
    # Check distinctness
    if len(set(tuple(p) for p in points)) != 12:
        return None
    # Check bounds
    for p in points:
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return None
    # Check not all collinear
    collinear = True
    for i in range(1, 12):
        dx = points[i][0] - points[0][0]
        dy = points[i][1] - points[0][1]
        for j in range(i + 1, 12):
            dx2 = points[j][0] - points[0][0]
            dy2 = points[j][1] - points[0][1]
            if dx * dy2 != dy * dx2:
                collinear = False
                break
        if not collinear:
            break
    if collinear:
        return None
    return _count_ordinary_lines(points)

def _get_lines_for_points(points):
    """Return dict mapping line key -> set of point indices on that line."""
    n = len(points)
    line_map = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            a = dy
            b = -dx
            c = a * x1 + b * y1
            g2 = gcd(abs(a), abs(b))
            if g2 > 0:
                a //= g2
                b //= g2
                c //= g2
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
                c = -c
            line_map[(a, b, c)].add(i)
            line_map[(a, b, c)].add(j)
    return line_map

def _count_ordinary_from_lines(line_map):
    """Count ordinary lines from a precomputed line_map."""
    return sum(1 for line, pts in line_map.items() if len(pts) == 2)

def _snap_candidates(points, i, line_map):
    """
    Generate candidate positions for point i by snapping to existing lines.
    For each line with >= 2 other points, find integer points on that line
    that are not already occupied.
    """
    n = len(points)
    candidates = set()
    for line_key, pts_on_line in line_map.items():
        if i in pts_on_line:
            pts_on_line = pts_on_line - {i}
        else:
            pass
        if len(pts_on_line) < 2:
            continue
        # Get the line equation a*x + b*y = c
        a, b, c = line_key
        # Find two points on this line to get direction
        pts_list = list(pts_on_line)
        if len(pts_list) < 2:
            continue
        p1 = points[pts_list[0]]
        p2 = points[pts_list[1]]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        g = gcd(abs(dx), abs(dy))
        if g > 0:
            dx //= g
            dy //= g
        # Direction vector is (dx, dy), but we need to be careful with sign
        # Actually, let's use the line equation directly
        # a*x + b*y = c, find integer solutions
        # We need gcd(a,b) | c for solutions to exist (should be true since points are on it)
        ga, gb = a, b
        gg = gcd(abs(ga), abs(gb))
        if gg == 0:
            continue
        ga_n = ga // gg
        gb_n = gb // gg
        c_n = c // gg
        if c_n * gg != c:
            continue  # No integer solution
        # Find one particular solution
        # ga_n * x + gb_n * y = c_n
        # Extended Euclidean
        def egcd(a, b):
            if b == 0:
                return (1, 0, a)
            x1, y1, g = egcd(b, a % b)
            return (y1, x1 - (a // b) * y1, g)
        
        x0, y0, g_val = egcd(ga_n, gb_n)
        if g_val != 1:
            continue
        # Particular solution: (x0 * c_n, y0 * c_n)
        px = x0 * c_n
        py = y0 * c_n
        # General solution: x = px + gb_n * t, y = py - ga_n * t
        # Try t values to get points near existing points
        existing = set(tuple(p) for j, p in enumerate(points) if j != i)
        # Try t values around 0 and around existing points
        t_range = range(-50, 51)
        for t in t_range:
            nx = px + gb_n * t
            ny = py - ga_n * t
            if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                if (nx, ny) not in existing:
                    candidates.add((nx, ny))
    return list(candidates)

def _generate_pg23_candidates():
    """
    Generate candidates based on PG(2, 3) structure.
    PG(2, 3) has 13 points. Removing one gives 12.
    We map these to integer coordinates.
    """
    candidates = []
    # Points in PG(2, 3) can be represented by (x, y, z) with x,y,z in {0,1,2} not all 0.
    # We pick 12 such points.
    # To map to affine plane, we can use a projection.
    # Let's try to map (x,y,z) -> (x/z, y/z) if z!=0, else handle infinity.
    # To keep integers, we can scale.
    
    # A known good configuration is the 3x4 grid, but let's try to generate
    # points that form many 3-point lines.
    
    # Try 4 lines of 3 points, intersecting at a common point P=(0,0).
    # Lines: y=0, y=x, y=2x, y=3x.
    # Points on y=0: (1,0), (2,0), (3,0)
    # Points on y=x: (1,1), (2,2), (3,3)
    # Points on y=2x: (1,2), (2,4), (3,6)
    # Points on y=3x: (1,3), (2,6), (3,9)
    # Plus P=(0,0). Total 13. Remove one?
    # If we remove P, we have 12 points.
    # Do they form many 3-point lines?
    # The original lines are broken (2 points each).
    # New lines might form.
    
    pts = [[1,0], [2,0], [3,0],
           [1,1], [2,2], [3,3],
           [1,2], [2,4], [3,6],
           [1,3], [2,6], [3,9]]
    candidates.append(pts)
    
    # Try 3 lines of 4 points, concurrent at P=(0,0).
    # y=0: (1,0)..(4,0)
    # y=x: (1,1)..(4,4)
    # y=2x: (1,2)..(4,8)
    # Plus P? Total 13. Remove P.
    pts2 = [[1,0], [2,0], [3,0], [4,0],
            [1,1], [2,2], [3,3], [4,4],
            [1,2], [2,4], [3,6], [4,8]]
    candidates.append(pts2)
    
    # Try 2 lines of 6 points, concurrent at P=(0,0).
    # y=0: (1,0)..(6,0)
    # y=x: (1,1)..(6,6)
    # Plus P? Total 13. Remove P.
    pts3 = [[1,0], [2,0], [3,0], [4,0], [5,0], [6,0],
            [1,1], [2,2], [3,3], [4,4], [5,5], [6,6]]
    candidates.append(pts3)
    
    return candidates

def construct():
    """
    Constructs 12 distinct integer points minimizing ordinary lines.
    """
    start_time = time.time()
    deadline = start_time + 145
    
    candidates = []
    
    # Standard grids
    candidates.append([[x, y] for x in range(4) for y in range(3)])
    candidates.append([[x, y] for x in range(3) for y in range(4)])
    
    # PG(2, 3) inspired
    candidates.extend(_generate_pg23_candidates())
    
    # Find best initial
    best_points = None
    best_count = float('inf')
    
    for pts in candidates:
        count = _try_config(pts)
        if count is not None and count < best_count:
            best_count = count
            best_points = [list(p) for p in pts]
    
    if best_points is None:
        best_points = [[i, 0] for i in range(11)] + [[0, 1]]
        best_count = _try_config(best_points)
        
    # Aggressive Local Search: Try to move a point to create a 3-point line
    # We identify lines with 2 points and try to move a 3rd point onto them.
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        line_map = _get_lines_for_points(best_points)
        
        # Find all ordinary lines (2 points)
        ordinary_lines = [line for line, pts in line_map.items() if len(pts) == 2]
        
        for line in ordinary_lines:
            if time.time() > deadline - 10:
                break
            a, b, c = line
            pts_on_line = list(line_map[line])
            p1_idx, p2_idx = pts_on_line
            p1 = best_points[p1_idx]
            p2 = best_points[p2_idx]
            
            # Try to move another point k onto this line
            for k in range(12):
                if k == p1_idx or k == p2_idx:
                    continue
                
                # Find a point on the line a*x + b*y = c that is close to best_points[k]
                # and is an integer.
                # We can parametrize the line.
                # Direction vector: (b, -a) normalized?
                # Let's just try moving k to a few integer points on the line.
                
                # Get direction
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                g = gcd(abs(dx), abs(dy))
                if g > 0:
                    dx //= g
                    dy //= g
                
                # Points on line: p1 + t * (dx, dy)
                # Try t values around the position of k
                # Project k onto the line?
                # Simpler: try t in a range.
                # Since coordinates are small, t is small.
                
                for t in range(-10, 11):
                    nx = p1[0] + t * dx
                    ny = p1[1] + t * dy
                    if abs(nx) > 10**9 or abs(ny) > 10**9:
                        continue
                    if (nx, ny) == tuple(best_points[p1_idx]) or (nx, ny) == tuple(best_points[p2_idx]):
                        continue
                    if any((nx, ny) == tuple(best_points[j]) for j in range(12) if j != k):
                        continue
                    
                    trial = [list(p) for p in best_points]
                    trial[k] = [nx, ny]
                    count = _try_config(trial)
                    if count is not None and count < best_count:
                        best_points = trial
                        best_count = count
                        improved = True
                        break # Restart loop with new config
                if improved:
                    break
            if improved:
                break
    
    return best_points
