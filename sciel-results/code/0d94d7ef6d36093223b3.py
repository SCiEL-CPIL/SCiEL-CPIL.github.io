import time
from itertools import combinations
from math import isqrt

_START = None

def _elapsed():
    global _START
    if _START is None:
        import time as _t
        _START = _t.time()
    return time.time() - _START

def _deadline_check(limit=140.0):
    return _elapsed() < limit

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _dist_sq(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

def _are_collinear(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) == 0

def _are_concyclic(p1, p2, p3, p4):
    # Determinant of the circle equation matrix
    # | x^2+y^2  x  y  1 |
    # | x1^2+y1^2 x1 y1 1 |
    # | x2^2+y2^2 x2 y2 1 |
    # | x3^2+y3^2 x3 y3 1 |
    # Using the property that 4 points are concyclic iff the cross ratio is real?
    # Or simply compute the determinant.
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    # Shift coordinates to p1 to reduce magnitude
    # Actually, just compute the determinant directly
    # To avoid large numbers, we can use the formula:
    # det = (x1^2+y1^2)(x2 y3 - x3 y2) - x1 (y1(x2 y3 - x3 y2) - y1(x2 y3 - x3 y2)) ... this is messy.
    
    # Let's use the circle equation coefficients D, E, F derived from p1, p2, p3.
    # x^2 + y^2 + Dx + Ey + F = 0
    # D, E, F are rational if points are integer.
    # We already have the logic in the original code, but let's make it robust.
    
    a1 = x1 - x2
    b1 = y1 - y2
    c1 = x2*x2 + y2*y2 - x1*x1 - y1*y1
    
    a2 = x1 - x3
    b2 = y1 - y3
    c2 = x3*x3 + y3*y3 - x1*x1 - y1*y1
    
    det = a1 * b2 - a2 * b1
    if det == 0:
        return False
    
    D_num = c1 * b2 - c2 * b1
    E_num = a1 * c2 - a2 * c1
    
    # Check p4
    # det * (x4^2 + y4^2) + D_num * x4 + E_num * y4 - det * (x1^2 + y1^2) - D_num * x1 - E_num * y1 == 0
    val = (det * (x4*x4 + y4*y4) + D_num * x4 + E_num * y4 
           - det * (x1*x1 + y1*y1) - D_num * x1 - E_num * y1)
    return val == 0

def _valid_set(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            dsq = _dist_sq(points[i], points[j])
            if dsq == 0:
                return False
            if not _is_square(dsq):
                return False
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _are_collinear(points[i], points[j], points[k]):
                    return False
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if _are_concyclic(points[i], points[j], points[k], points[l]):
                        return False
    return True

def construct():
    global _START
    _START = time.time()
    deadline = 140.0
    
    # Known 5-point set
    fallback = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best = fallback
    best_size = 5
    
    # Try to find a 6th point
    # Search in a larger box
    # The 5 points are roughly in [-60, 20] x [0, 70]
    # Try a box of size 300x300 centered at (0, 30)
    
    found_6 = False
    for x in range(-150, 151):
        if not _deadline_check(deadline):
            break
        for y in range(-30, 151):
            if not _deadline_check(deadline):
                break
            p = (x, y)
            # Check distance to all 5
            ok = True
            for sp in fallback:
                dsq = _dist_sq(p, sp)
                if dsq == 0 or not _is_square(dsq):
                    ok = False
                    break
            if not ok:
                continue
            
            new_pts = fallback + [p]
            # Check collinearity
            ok = True
            for i in range(6):
                if not ok: break
                for j in range(i+1, 6):
                    if not ok: break
                    for k in range(j+1, 6):
                        if _are_collinear(new_pts[i], new_pts[j], new_pts[k]):
                            ok = False
                            break
            if not ok:
                continue
            
            # Check concyclicity
            ok = True
            for i in range(6):
                if not ok: break
                for j in range(i+1, 6):
                    if not ok: break
                    for k in range(j+1, 6):
                        if not ok: break
                        for l in range(k+1, 6):
                            if _are_concyclic(new_pts[i], new_pts[j], new_pts[k], new_pts[l]):
                                ok = False
                                break
            if ok:
                best = new_pts
                best_size = 6
                found_6 = True
                break
        if found_6:
            break
            
    # If 6 found, try 7? Unlikely.
    
    D = 1
    points_out = [[[x, 1], [y, 1]] for x, y in best]
    return {'D': D, 'points': points_out}
