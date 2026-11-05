"""Rewritten program: search for integral point sets in general position."""
import time
from math import isqrt
from fractions import Fraction

_START = time.time()
_DEADLINE = 135.0  # safety margin before 150s kill

def _elapsed():
    return time.time() - _START

def _time_left():
    return _DEADLINE - _elapsed()

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _sqdist_int(p, q, D):
    """Squared distance between two points (x,y) in the (x, y*sqrt(D)) space.
    Returns (num, den) as a rational or just an int if D=1 and coords are int.
    Actually for integer coords and D=1, it's just an integer."""
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + D * dy * dy

def _check_rational_distances(points, D):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            d2 = _sqdist_int(points[i], points[j], D)
            if not _is_square(d2):
                return False
    return True

def _check_no_three_collinear(points, D):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                # Collinear iff (x2-x1)*(y3-y1) == (x3-x1)*(y2-y1)
                # This works for the (x, y*sqrt(D)) representation since sqrt(D) cancels
                if (x2 - x1) * y3 == (x3 - x1) * y2:
                    return False
    return True

def _check_no_four_concyclic(points, D):
    """Check no four points are concyclic in the Euclidean plane (x, y*sqrt(D))."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
                    x1, y1 = p1[0], p1[1]
                    x2, y2 = p2[0], p2[1]
                    x3, y3 = p3[0], p3[1]
                    x4, y4 = p4[0], p4[1]
                    
                    if D == 1:
                        # Standard circle: x^2 + y^2 + ax + by + c = 0
                        # Use 4x4 determinant
                        # |x^2+y^2, x, y, 1| for 4 points
                        a1 = x1*x1 + y1*y1
                        a2 = x2*x2 + y2*y2
                        a3 = x3*x3 + y3*y3
                        a4 = x4*x4 + y4*y4
                        # Determinant of:
                        # a1 x1 y1 1
                        # a2 x2 y2 1
                        # a3 x3 y3 1
                        # a4 x4 y4 1
                        # Use the formula for 4x4 determinant
                        # Let's compute it directly
                        # M = [[a1,x1,y1,1],[a2,x2,y2,1],[a3,x3,y3,1],[a4,x4,y4,1]]
                        # det = a1*(x2*(y3*1-y4*1) - y2*(x3*1-x4*1) + 1*(x3*y4-x4*y3))
                        #     - x1*(a2*(y3-y4) - y2*(a3-a4) + 1*(a3*y4-a4*y3))
                        #     + y1*(a2*(x3-x4) - x2*(a3-a4) + 1*(a3*x4-a4*x3))
                        #     - 1*(a2*(x3*y4-x4*y3) - x2*(a3*y4-a4*y3) + y2*(a3*x4-a4*x3))
                        # This is getting messy. Let me use a simpler method.
                        # Check if 4 points are concyclic: 
                        # The circle through p1,p2,p3 is unique (if not collinear).
                        # Check if p4 lies on it.
                        # Solve for a,b,c from:
                        # a*x1 + b*y1 + c = -(x1^2+y1^2)
                        # a*x2 + b*y2 + c = -(x2^2+y2^2)
                        # a*x3 + b*y3 + c = -(x3^2+y3^2)
                        # Subtract 1 from 2: a*(x2-x1) + b*(y2-y1) = -(x2^2+y2^2-x1^2-y1^2)
                        # Subtract 1 from 3: a*(x3-x1) + b*(y3-y1) = -(x3^2+y3^2-x1^2-y1^2)
                        # Solve 2x2 system.
                        A = x2 - x1
                        B = y2 - y1
                        C = x3 - x1
                        Dmat = y3 - y1
                        E = -(x2*x2 + y2*y2 - x1*x1 - y1*y1)
                        F = -(x3*x3 + y3*y3 - x1*x1 - y1*y1)
                        det = A * Dmat - B * C
                        if det == 0:
                            # p1,p2,p3 collinear, can't form a circle
                            continue
                        # a = (E*Dmat - F*B) / det
                        # b = (A*F - C*E) / det
                        a_num = E * Dmat - F * B
                        b_num = A * F - C * E
                        # c = -(x1^2+y1^2) - a*x1 - b*y1
                        # c = -(x1^2+y1^2) - a_num/det * x1 - b_num/det * y1
                        # c = (-det*(x1^2+y1^2) - a_num*x1 - b_num*y1) / det
                        c_num = -det * (x1*x1 + y1*y1) - a_num * x1 - b_num * y1
                        # Check p4: a*x4 + b*y4 + c = -(x4^2+y4^2)
                        # a_num/det * x4 + b_num/det * y4 + c_num/det = -(x4^2+y4^2)
                        # a_num*x4 + b_num*y4 + c_num = -det*(x4^2+y4^2)
                        if a_num * x4 + b_num * y4 + c_num == -det * (x4*x4 + y4*y4):
                            return False
                    else:
                        # D > 1: circle in (x, y*sqrt(D)) plane
                        # x^2 + D*y^2 + a*x + b*y*sqrt(D) + c = 0
                        # (x^2 + D*y^2 + a*x + c) + b*y*sqrt(D) = 0
                        # For this to hold, if any y != 0, then b = 0
                        # and x^2 + D*y^2 + a*x + c = 0 for all 4 points.
                        if y1 != 0 or y2 != 0 or y3 != 0 or y4 != 0:
                            # b = 0, check x^2 + D*y^2 + a*x + c = 0
                            V1 = x1*x1 + D*y1*y1
                            V2 = x2*x2 + D*y2*y2
                            V3 = x3*x3 + D*y3*y3
                            V4 = x4*x4 + D*y4*y4
                            # a*x1 + c = -V1
                            # a*x2 + c = -V2
                            # a*(x2-x1) = V1 - V2
                            dx = x2 - x1
                            if dx != 0:
                                if (V1 - V2) % dx != 0:
                                    continue
                                a_val = (V1 - V2) // dx
                                c_val = -V1 - a_val * x1
                                if x3*x3 + D*y3*y3 + a_val*x3 + c_val != 0:
                                    continue
                                if x4*x4 + D*y4*y4 + a_val*x4 + c_val != 0:
                                    continue
                                return False
                            else:
                                # x1 == x2, need V1 == V2, i.e., y1^2 == y2^2
                                if V1 != V2:
                                    continue
                                # Try with points 1 and 3
                                dx2 = x3 - x1
                                if dx2 != 0:
                                    if (V1 - V3) % dx2 != 0:
                                        continue
                                    a_val = (V1 - V3) // dx2
                                    c_val = -V1 - a_val * x1
                                    if x2*x2 + D*y2*y2 + a_val*x2 + c_val != 0:
                                        continue
                                    if x4*x4 + D*y4*y4 + a_val*x4 + c_val != 0:
                                        continue
                                    return False
                        # else: all y=0, can't have 4 distinct points on a circle (at most 2)
    return True

def _validate(points, D):
    n = len(points)
    if n < 3:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            if points[i] == points[j]:
                return False
    if not _check_rational_distances(points, D):
        return False
    if not _check_no_three_collinear(points, D):
        return False
    if not _check_no_four_concyclic(points, D):
        return False
    return True

def _try_add_point(points, D, candidate):
    new_points = points + [candidate]
    return _validate(new_points, D)

def construct():
    # Known 5-point integral point set (D=1)
    # This is a well-known configuration
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    D = 1
    
    if not _validate(base_points, D):
        # Fallback to a simpler 3-point set
        base_points = [(0, 0), (3, 4), (0, 5)]
        if not _validate(base_points, D):
            base_points = [(0, 0), (1, 0), (0, 1)]
    
    best = list(base_points)
    
    # Search for additional points
    # Strategy: try to add points one at a time
    # For efficiency, only consider points at integer distance from all existing points
    
    search_range = 300
    max_points_found = len(best)
    
    # Try multiple rounds of adding points
    for round_num in range(20):
        if _time_left() < 15:
            break
        if len(best) >= 8:
            break
            
        added = False
        # Expand search range each round
        current_range = search_range + round_num * 50
        
        for x in range(-current_range, current_range + 1):
            if _time_left() < 10:
                break
            for y in range(-current_range, current_range + 1):
                if _time_left() < 5:
                    break
                candidate = (x, y)
                
                # Quick check: integer distance to all existing points
                ok = True
                for p in best:
                    d2 = (x - p[0])**2 + (y - p[1])**2
                    if not _is_square(d2):
                        ok = False
                        break
                if not ok:
                    continue
                
                # Full validation
                if _try_add_point(best, D, candidate):
                    best.append(candidate)
                    added = True
                    max_points_found = len(best)
                    if len(best) >= 8:
                        break
            if len(best) >= 8:
                break
        
        if not added:
            # No more points found in this range, try a different approach
            # Try shifting the entire configuration
            # Or try a different base configuration
            break
    
    # If we have fewer than 6 points, try a direct search for 6-point sets
    if len(best) < 6 and _time_left() > 30:
        # Try known 6-point configurations or search more broadly
        # For now, try a different base
        alt_bases = [
            [(0, 0), (5, 0), (0, 5), (5, 5), (13, 16), (16, 13)],  # May not be valid
            [(0, 0), (3, 4), (4, 3), (15, 8), (8, 15), (17, 0)],  # May not be valid
        ]
        for alt in alt_bases:
            if _validate(alt, D):
                if len(alt) > len(best):
                    best = list(alt)
                break
    
    # Final: try to add more points to the best set found
    if len(best) < 8 and _time_left() > 20:
        for x in range(-500, 501):
            if _time_left() < 10:
                break
            for y in range(-500, 501):
                if _time_left() < 5:
                    break
                candidate = (x, y)
                ok = True
                for p in best:
                    d2 = (x - p[0])**2 + (y - p[1])**2
                    if not _is_square(d2):
                        ok = False
                        break
                if not ok:
                    continue
                if _try_add_point(best, D, candidate):
                    best.append(candidate)
                    if len(best) >= 8:
                        break
            if len(best) >= 8:
                break
    
    return {'D': D, 'points': [[[x, 1], [y, 1]] for x, y in best]}
