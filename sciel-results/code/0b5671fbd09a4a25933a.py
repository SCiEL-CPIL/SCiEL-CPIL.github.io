"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def _count_ordinary(pts):
    """Count lines containing exactly 2 of the given points."""
    n = len(pts)
    if n < 2:
        return 0
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2
            g = gcd(gcd(abs(a), abs(b)), abs(c))
            if g > 0:
                a //= g
                b //= g
                c //= g
            # Canonical form: first nonzero of (a,b) positive
            if a < 0 or (a == 0 and b < 0):
                a, b, c = -a, -b, -c
            key = (a, b, c)
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    return sum(1 for v in line_map.values() if len(v) == 2)

def _is_valid(pts):
    """Check 12 distinct integer points, |x|,|y| <= 1e9, not all collinear."""
    if len(pts) != 12:
        return False
    seen = set()
    for p in pts:
        if not isinstance(p[0], int) or not isinstance(p[1], int):
            return False
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return False
        key = (p[0], p[1])
        if key in seen:
            return False
        seen.add(key)
    # Check not all collinear
    if len(pts) < 3:
        return True
    x0, y0 = pts[0]
    x1, y1 = pts[1]
    for i in range(2, len(pts)):
        xi, yi = pts[i]
        if (xi - x0) * (y1 - y0) != (yi - y0) * (x1 - x0):
            return True
    return False

def construct():
    # Deterministic construction based on 3 parallel lines with 4 points each
    # This configuration is known to have few ordinary lines
    # Line 1: y=0, x=0,1,2,3
    # Line 2: y=1, x=0,1,2,3
    # Line 3: y=2, x=0,1,2,3
    # This creates a 3x4 grid which has 6 ordinary lines (the vertical lines)
    
    # Try the 3x4 grid configuration
    grid_pts = [[x, y] for x in range(4) for y in range(3)]
    if _is_valid(grid_pts):
        score = _count_ordinary(grid_pts)
        # The 3x4 grid has 4 vertical lines (each with 3 points) and 3 horizontal lines (each with 4 points)
        # Plus diagonal lines. Let's count:
        # Vertical lines: x=0,1,2,3 -> 4 lines with 3 points each (not ordinary)
        # Horizontal lines: y=0,1,2 -> 3 lines with 4 points each (not ordinary)
        # Diagonal lines: need to count lines with exactly 2 points
        # For a 3x4 grid, the ordinary lines are the "short" diagonals
        # Actually, let's just compute it
        if score <= 6:
            return grid_pts
    
    # Try alternative: 4x3 grid (transposed)
    grid_pts2 = [[x, y] for x in range(3) for y in range(4)]
    if _is_valid(grid_pts2):
        score2 = _count_ordinary(grid_pts2)
        if score2 <= 6:
            return grid_pts2
    
    # If neither works well, use a more structured approach
    # Try points on a parabola y = x^2
    parabola_pts = [[x, x*x] for x in range(12)]
    if _is_valid(parabola_pts):
        score3 = _count_ordinary(parabola_pts)
        if score3 < 6:
            return parabola_pts
    
    # Fallback: return the grid configuration
    return grid_pts
# EVOLVE-BLOCK-END
