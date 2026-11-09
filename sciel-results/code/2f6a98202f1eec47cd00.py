"""
Strategy: 
1. Explore specific high-symmetry configurations known to have low ordinary line counts.
2. Specifically, test configurations derived from:
   a) 3 lines with 4 points each (if possible without overlap issues)
   b) 2 lines with 6 points each (parallel or intersecting)
   c) Grids with specific affine transformations
   d) Conic sections (parabola y=x^2) with a point at infinity or center
3. Use a fast exact counter for these specific structures.
4. Perform a limited local search only if the initial configurations don't hit the lower bound (6).
"""

import time
import math
import random
from itertools import combinations

def _line_key(p1, p2):
    """Compute normalized line key (a, b, c) for line through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    a = dy
    b = -dx
    c = -a * x1 - b * y1
    g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
    if g > 0:
        a //= g
        b //= g
        c //= g
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a = -a
        b = -b
        c = -c
    return (a, b, c)

def _count_ordinary(points):
    """Count the number of lines containing exactly 2 of the given points."""
    n = len(points)
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in lines:
                lines[key] = 0
            lines[key] += 1
    
    ordinary = 0
    for key, cnt in lines.items():
        if cnt == 2:
            ordinary += 1
    return ordinary

def _valid(points):
    """Check if points are valid: 12 distinct integer points, |x|,|y| <= 10^9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for p in points:
        if not isinstance(p, list) or len(p) != 2:
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
    if len(points) < 3:
        return True
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    if cross == 0:
        for i in range(3, 12):
            xi, yi = points[i]
            cross_i = (x2 - x1) * (yi - y1) - (y2 - y1) * (xi - x1)
            if cross_i != 0:
                return True
        return False
    return True

def _generate_config_3_lines_4_pts():
    """
    Try to construct 12 points on 3 lines, 4 points per line.
    If the lines intersect at a common point, we need to be careful about distinctness.
    Let's use 3 lines forming a triangle, with 4 points on each side.
    Vertices are shared? No, points must be distinct.
    If we put 4 points on line 1, 4 on line 2, 4 on line 3.
    Total 12 points.
    If they don't share vertices, we have 3 lines of 4.
    Ordinary lines: 
    Any pair on same line is non-ordinary.
    Pairs on different lines define a new line.
    If that new line contains a 3rd point, it's non-ordinary.
    This is the "3x4" grid structure essentially if the lines are parallel? No, 3 lines of 4.
    Let's try a grid 3x4.
    """
    # 3 rows, 4 columns
    # Points: (i, j) for i in 0..2, j in 0..3
    # Rows: 3 lines of 4 points.
    # Cols: 4 lines of 3 points.
    # Diagonals?
    pts = [[i, j] for i in range(3) for j in range(4)]
    return pts

def _generate_config_2_lines_6_pts():
    """
    2 parallel lines, 6 points each.
    """
    pts = []
    for i in range(6):
        pts.append([i, 0])
        pts.append([i, 1])
    return pts

def _generate_config_parabola():
    """
    Points on parabola y = x^2.
    x = -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5 (11 points)
    Plus one point at (0, -1) or similar to break symmetry?
    Or just 12 points on parabola?
    If 12 points on parabola, any 3 are non-collinear? No, 3 points on parabola are never collinear.
    So every pair defines an ordinary line?
    Wait, if all 12 are on a conic, no 3 are collinear.
    Then ALL C(12,2)=66 lines are ordinary. That's bad.
    We want MAXIMUM collinearity.
    """
    return None

def _generate_config_hexagonal():
    """
    Hexagonal lattice points.
    """
    # Center (0,0)
    # 6 neighbors
    # 6 at distance 2
    pts = [[0, 0]]
    # Neighbors in hex grid (approximated in cartesian)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) -- this is a hexagon
    # Let's use a standard hex grid coordinate transform
    # x = 2*i + j, y = j
    # i, j in [-1, 1] gives 9 points.
    # Add 3 more?
    # Let's try a 3x4 grid again, it's robust.
    
    # Let's try a configuration based on the "orchard problem"
    # Max points on lines.
    # For 12 points, max lines of 3?
    # A 3x4 grid has:
    # 3 rows of 4
    # 4 cols of 3
    # 2 diagonals of 3? (0,0)-(1,1)-(2,2) and (0,2)-(1,1)-(2,0)? No, (0,0)-(1,1)-(2,2) is 3 pts. (0,2)-(1,1)-(2,0) is 3 pts.
    # Are there more?
    # (0,0)-(2,1)? No.
    # Let's count ordinary lines for 3x4 grid.
    # Total pairs: 66.
    # Non-ordinary pairs:
    # Rows: 3 lines * C(4,2)=6 = 18 pairs.
    # Cols: 4 lines * C(3,2)=3 = 12 pairs.
    # Diagonals: 2 lines * C(3,2)=3 = 6 pairs.
    # Total non-ordinary pairs = 18+12+6 = 36.
    # Ordinary pairs = 66 - 36 = 30.
    # Wait, ordinary LINES count, not pairs.
    # A line with 3 points accounts for 3 pairs, but is 0 ordinary lines.
    # A line with 2 points is 1 ordinary line.
    # So we need to count lines with exactly 2 points.
    
    return None

def construct():
    deadline = time.time() + 115
    
    best_points = None
    best_count = float('inf')
    
    # Candidate 1: 3x4 Grid
    pts1 = [[i, j] for i in range(3) for j in range(4)]
    if _valid(pts1):
        c1 = _count_ordinary(pts1)
        if c1 < best_count:
            best_count = c1
            best_points = [p[:] for p in pts1]
    
    # Candidate 2: 4x3 Grid (transposed)
    pts2 = [[i, j] for i in range(4) for j in range(3)]
    if _valid(pts2):
        c2 = _count_ordinary(pts2)
        if c2 < best_count:
            best_count = c2
            best_points = [p[:] for p in pts2]
    
    # Candidate 3: 2x6 Grid
    pts3 = [[i, j] for i in range(2) for j in range(6)]
    if _valid(pts3):
        c3 = _count_ordinary(pts3)
        if c3 < best_count:
            best_count = c3
            best_points = [p[:] for p in pts3]
    
    # Candidate 4: Affine transformed 3x4 grid
    # Try a few random affine transforms to see if we can improve
    # (x,y) -> (a*x+b*y, c*x+d*y)
    # We want to preserve integer coordinates.
    # Let's try a shear.
    for shear in range(-3, 4):
        pts4 = [[i + shear*j, j] for i in range(3) for j in range(4)]
        if _valid(pts4):
            c4 = _count_ordinary(pts4)
            if c4 < best_count:
                best_count = c4
                best_points = [p[:] for p in pts4]
    
    # Candidate 5: "Pappus" like
    # Two lines y=0, y=1
    # Points on y=0: 0,1,2,3,4,5
    # Points on y=1: 0,1,2,3,4,5
    # This is the 2x6 grid.
    
    # Candidate 6: 3 lines of 4 points, not a grid.
    # Line 1: y=0, x=0,1,2,3
    # Line 2: y=1, x=0,1,2,3
    # Line 3: x=0, y=0,1,2,3 -> (0,0) and (0,1) are shared.
    # We need 12 DISTINCT points.
    # So we can't just share vertices easily unless we count them once.
    # If we have 3 lines, each with 4 points, and they intersect pairwise at 3 points.
    # Total points = 4+4+4 - 3 (intersections) = 9.
    # We need 12. So we need 3 more points.
    # This gets complicated.
    
    # Let's try a local search on the best candidate found so far.
    if best_points is None:
        best_points = [[i, 0] for i in range(11)] + [[0, 1]]
        best_count = _count_ordinary(best_points)
        
    # Simple local search
    current_points = best_points
    current_count = best_count
    
    # Try moving one point to break an ordinary line or create a new collinearity
    # We only do a few iterations to stay fast.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(12):
            x, y = current_points[i]
            # Try small moves
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if abs(nx) > 10**9 or abs(ny) > 10**9:
                        continue
                    new_pts = [p[:] for p in current_points]
                    new_pts[i] = [nx, ny]
                    if _valid(new_pts):
                        new_count = _count_ordinary(new_pts)
                        if new_count < current_count:
                            current_points = new_pts
                            current_count = new_count
                            improved = True
                            if current_count <= 6:
                                return current_points
    
    return current_points
