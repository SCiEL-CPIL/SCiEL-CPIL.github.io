"""Search for integral point sets in general position."""
import math
import time
from itertools import combinations

def _is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def _sqdist(p, q):
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2

def _collinear(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) == 0

def _concyclic(a, b, c, d):
    # Check if 4 points are concyclic using determinant
    ax, ay = a; bx, by = b; cx, cy = c; dx, dy = d
    # Circle through a,b,c: Ax^2+Ay^2+Bx+Cy+D=0
    # Subtract to get linear system for B,C,D
    # bx^2+by^2+bx*B+by*C+D = ax^2+ay^2+ax*B+ay*C+D
    # (bx-ax)*B + (by-ay)*C = ax^2+ay^2 - bx^2-by^2
    # Similarly for c
    A1 = bx - ax; B1 = by - ay; C1 = ax*ax+ay*ay - bx*bx-by*by
    A2 = cx - ax; B2 = cy - ay; C2 = ax*ax+ay*ay - cx*cx-cy*cy
    det = A1*B2 - A2*B1
    if det == 0:
        return True  # a,b,c collinear, any d works (degenerate)
    B = (C1*B2 - C2*B1) // det if (C1*B2 - C2*B1) % det == 0 else (C1*B2 - C2*B1) / det
    C = (A1*C2 - A2*C1) // det if (A1*C2 - A2*C1) % det == 0 else (A1*C2 - A2*C1) / det
    D = -(ax*ax+ay*ay+ax*B+ay*C)
    # Check d
    return dx*dx+dy*dy+dx*B+dy*C+D == 0

def _verify(points):
    n = len(points)
    if n < 3:
        return False
    # Check all pairs have integer distance
    for i in range(n):
        for j in range(i+1, n):
            d2 = _sqdist(points[i], points[j])
            if d2 == 0:
                return False
            if not _is_square(d2):
                return False
    # No three collinear
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _collinear(points[i], points[j], points[k]):
                    return False
    # No four concyclic
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for m in range(k+1, n):
                    if _concyclic(points[i], points[j], points[k], points[m]):
                        return False
    return True

def _try_add(point, pts):
    """Try adding point to pts, return True if valid."""
    new_pts = pts + [point]
    n = len(new_pts)
    # Check distances from new point to all others
    for i in range(n-1):
        d2 = _sqdist(point, new_pts[i])
        if d2 == 0 or not _is_square(d2):
            return False
    # Check no three collinear involving the new point
    for i in range(n-1):
        for j in range(i+1, n-1):
            if _collinear(point, new_pts[i], new_pts[j]):
                return False
    # Check no four concyclic involving the new point
    for i in range(n-1):
        for j in range(i+1, n-1):
            for k in range(j+1, n-1):
                if _concyclic(point, new_pts[i], new_pts[j], new_pts[k]):
                    return False
    return True

def _search(start_pts, max_radius=200, target_size=7):
    """Search for a larger set by trying to add points."""
    pts = list(start_pts)
    current_size = len(pts)
    
    # Try to add one point at a time
    for radius in range(1, max_radius):
        if len(pts) >= target_size:
            break
        # Generate candidate points on a grid
        for x in range(-radius, radius+1):
            for y in range(-radius, radius+1):
                candidate = (x, y)
                # Skip if already in set
                if candidate in pts:
                    continue
                if _try_add(candidate, pts):
                    pts.append(candidate)
                    if len(pts) >= target_size:
                        return pts
        if len(pts) >= target_size:
            return pts
    return pts

def construct():
    deadline = time.time() + 130  # Safety margin before 150s kill
    
    # Start with the known 5-point set
    base_pts = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    if not _verify(base_pts):
        # Fallback: simple triangle
        base_pts = [(0, 0), (3, 4), (0, 6)]
    
    best_pts = base_pts
    best_size = len(base_pts)
    
    # Try to grow the set
    # Strategy 1: Try to add points to the 5-point set
    t0 = time.time()
    grown = _search(base_pts, max_radius=100, target_size=7)
    if len(grown) > best_size and _verify(grown):
        best_pts = grown
        best_size = len(grown)
    
    # Strategy 2: Try different starting configurations
    if time.time() < deadline:
        # Try a smaller base and grow
        small_sets = [
            [(0,0), (3,4), (5,12)],
            [(0,0), (4,3), (12,5)],
            [(0,0), (5,0), (0,12)],
            [(0,0), (9,12), (15,8)],
            [(0,0), (7,24), (24,7)],
            [(0,0), (8,15), (15,8)],
            [(0,0), (12,5), (5,12)],
            [(0,0), (20,21), (21,20)],
            [(0,0), (9,40), (40,9)],
            [(0,0), (11,60), (60,11)],
            [(0,0), (28,45), (45,28)],
            [(0,0), (33,56), (56,33)],
            [(0,0), (48,55), (55,48)],
            [(0,0), (13,84), (84,13)],
            [(0,0), (36,77), (77,36)],
            [(0,0), (65,72), (72,65)],
            [(0,0), (25,60), (60,25)],
            [(0,0), (40,75), (75,40)],
            [(0,0), (15,80), (80,15)],
            [(0,0), (35,84), (84,35)],
        ]
        
        for s in small_sets:
            if time.time() >= deadline - 5:
                break
            if not _verify(s):
                continue
            grown = _search(s, max_radius=80, target_size=7)
            if len(grown) > best_size and _verify(grown):
                best_pts = grown
                best_size = len(grown)
            if best_size >= 7:
                break
    
    # Strategy 3: Try even smaller bases with larger search radius
    if time.time() < deadline - 10 and best_size < 6:
        for a in range(1, 30):
            if time.time() >= deadline - 10:
                break
            for b in range(1, 30):
                if a*a + b*b == 0:
                    continue
                # Generate Pythagorean triple points
                # Point at distance sqrt(a^2+b^2) from origin
                p1 = (a, b)
                # Find another point at integer distance from both origin and p1
                # Try systematic search
                best_3 = None
                for x in range(-100, 101):
                    for y in range(-100, 101):
                        if (x,y) == (0,0) or (x,y) == (a,b):
                            continue
                        d1 = x*x + y*y
                        d2 = (x-a)*(x-a) + (y-b)*(y-b)
                        if _is_square(d1) and _is_square(d2) and d1 > 0 and d2 > 0:
                            if not _collinear((0,0), (a,b), (x,y)):
                                best_3 = [(0,0), (a,b), (x,y)]
                                break
                    if best_3:
                        break
                if best_3:
                    grown = _search(best_3, max_radius=60, target_size=7)
                    if len(grown) > best_size and _verify(grown):
                        best_pts = grown
                        best_size = len(grown)
                    if best_size >= 7:
                        break
    
    # Final verification
    if not _verify(best_pts):
        best_pts = [(0,0), (3,4), (0,6)]
    
    # Check size
    if len(best_pts) < 3:
        best_pts = [(0,0), (3,4), (0,6)]
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_pts]}
