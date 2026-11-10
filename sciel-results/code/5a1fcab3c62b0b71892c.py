import time
import math
import random
from fractions import Fraction

_START = time.time()
_DEADLINE = 115.0  # hard wall-clock budget (safety margin vs 120s)

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _check_rational_distance(p1, p2, D):
    # p1, p2 are (x_num, x_den, y_num, y_den)
    # distance squared = (x1 - x2)^2 + D*(y1 - y2)^2
    dx_num = p1[0] * p2[1] - p2[0] * p1[1]
    dx_den = p1[1] * p2[1]
    dy_num = p1[2] * p2[3] - p2[2] * p1[3]
    dy_den = p1[3] * p2[3]
    
    # dist_sq = (dx_num/dx_den)^2 + D * (dy_num/dy_den)^2
    # = dx_num^2 / dx_den^2 + D * dy_num^2 / dy_den^2
    # = (dx_num^2 * dy_den^2 + D * dy_num^2 * dx_den^2) / (dx_den^2 * dy_den^2)
    
    num = dx_num * dx_num * dy_den * dy_den + D * dy_num * dy_num * dx_den * dx_den
    den = dx_den * dx_den * dy_den * dy_den
    
    if den == 0:
        return False
    return _is_square(num)

def _check_collinear(p1, p2, p3):
    # p1, p2, p3 are (x_num, x_den, y_num, y_den)
    # (x2-x1)(y3-y1) - (y2-y1)(x3-x1) == 0
    # Let's compute in exact rationals or cross multiply
    # x2-x1 = (x2n*x1d - x1n*x2d) / (x1d*x2d)
    # y3-y1 = (y3n*y1d - y1n*y3d) / (y1d*y3d)
    # Product = ... / (x1d*x2d*y1d*y3d)
    # Similarly for other term.
    # Numerator of difference must be 0.
    
    x1n, x1d, y1n, y1d = p1
    x2n, x2d, y2n, y2d = p2
    x3n, x3d, y3n, y3d = p3
    
    # To avoid huge denominators, we can just compute the determinant of:
    # x1 y1 1
    # x2 y2 1
    # x3 y3 1
    # But x,y are rationals.
    # Multiply row i by x_id * y_id? No, just clear denominators.
    # Let X_i = x_i * LCM? No, just use the property that det is 0.
    # det = x1(y2-y3) - y1(x2-x3) + (x2y3 - x3y2) = 0
    
    # Let's compute:
    # term1 = x1 * (y2 - y3)
    # term2 = -y1 * (x2 - x3)
    # term3 = x2 * y3 - x3 * y2
    
    # x1 = x1n/x1d, etc.
    # This is getting messy with raw integers.
    # Easier: convert to Fraction for the check?
    # Performance might be okay for small N (<= 12).
    
    x1 = Fraction(x1n, x1d)
    y1 = Fraction(y1n, y1d)
    x2 = Fraction(x2n, x2d)
    y2 = Fraction(y2n, y2d)
    x3 = Fraction(x3n, x3d)
    y3 = Fraction(y3n, y3d)
    
    return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

def _check_concyclic(p1, p2, p3, p4):
    # Convert to Fraction tuples
    pts = []
    for p in (p1, p2, p3, p4):
        x = Fraction(p[0], p[1])
        y = Fraction(p[2], p[3])
        pts.append((x, y))
    
    x1, y1 = pts[0]
    x2, y2 = pts[1]
    x3, y3 = pts[2]
    x4, y4 = pts[3]
    
    # Determinant of:
    # x^2+y^2  x  y  1
    # ...
    # Subtract row 4 from 1,2,3
    x4sq = x4*x4 + y4*y4
    r1 = (x1*x1 + y1*y1 - x4sq, x1 - x4, y1 - y4)
    r2 = (x2*x2 + y2*y2 - x4sq, x2 - x4, y2 - y4)
    r3 = (x3*x3 + y3*y3 - x4sq, x3 - x4, y3 - y4)
    
    det = (r1[0] * (r2[1]*r3[2] - r2[2]*r3[1]) - 
           r1[1] * (r2[0]*r3[2] - r2[2]*r3[0]) + 
           r1[2] * (r2[0]*r3[1] - r2[1]*r3[0]))
    return det == 0

def _is_valid_set(points, D):
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            if not _check_rational_distance(points[i], points[j], D):
                return False
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _check_collinear(points[i], points[j], points[k]):
                    return False
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if _check_concyclic(points[i], points[j], points[k], points[l]):
                        return False
    return True

def _try_add_point(current_points, candidate, D):
    # candidate is (x_num, x_den, y_num, y_den)
    # Check distance to all existing
    for p in current_points:
        if not _check_rational_distance(candidate, p, D):
            return False
    # Check collinearity
    n = len(current_points)
    for i in range(n):
        for j in range(i+1, n):
            if _check_collinear(candidate, current_points[i], current_points[j]):
                return False
    # Check concyclicity
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _check_concyclic(candidate, current_points[i], current_points[j], current_points[k]):
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Start with the known 5-point set
    # (0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)
    # Format: [x_num, x_den, y_num, y_den]
    base_points = [
        [0, 1, 0, 1], 
        [-56, 1, 33, 1], 
        [-16, 1, 63, 1], 
        [0, 1, 33, 1], 
        [16, 1, 63, 1]
    ]
    D = 1
    
    # Verify base points
    if not _is_valid_set(base_points, D):
        base_points = [
            [0, 1, 0, 1],
            [3, 1, 4, 1],
            [0, 1, 4, 1]
        ]
        D = 1

    current_points = base_points[:]
    
    # Search for more points
    # We will generate candidates from a grid of small integers
    # and also try to generate candidates using rational parametrization of circles
    # passing through pairs of existing points.
    
    # Grid search first
    max_coord = 100
    candidates = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            candidates.append([x, 1, y, 1])
    
    # Remove existing points from candidates
    current_set = set(tuple(p) for p in current_points)
    candidates = [c for c in candidates if tuple(c) not in current_set]
    
    added_any = True
    while added_any and time.time() < deadline:
        added_any = False
        
        # Filter candidates
        valid_candidates = []
        for p in candidates:
            if _try_add_point(current_points, p, D):
                valid_candidates.append(p)
        
        if valid_candidates:
            # Pick the one with smallest norm
            valid_candidates.sort(key=lambda p: p[0]*p[0]*p[1]*p[1] + p[2]*p[2]*p[3]*p[3])
            new_point = valid_candidates[0]
            current_points.append(new_point)
            added_any = True
            # Remove from candidates
            idx = candidates.index(new_point)
            candidates.pop(idx)
            
            # Check deadline
            if time.time() > deadline - 5.0:
                break
        
        # If no integer points found, try to generate rational points
        # This is more complex, so for now we stick to integers if possible.
        # But the task allows rationals.
        # For now, if we are stuck, we might try a different D or rationals.
        # Given time constraints, let's try to expand the grid or use rationals.
        
        # If we haven't found any new integer points, let's try some rational candidates
        # derived from circle intersections or just random rationals with small denominators.
        if not added_any:
             # Try a few random rationals
             found = False
             for _ in range(1000):
                 if time.time() > deadline - 5.0:
                     break
                 x_num = random.randint(-10, 10)
                 x_den = random.randint(1, 5)
                 y_num = random.randint(-10, 10)
                 y_den = random.randint(1, 5)
                 cand = [x_num, x_den, y_num, y_den]
                 # Normalize
                 g1 = _gcd(x_num, x_den)
                 g2 = _gcd(y_num, y_den)
                 if g1 > 0:
                     cand[0] //= g1; cand[1] //= g1
                 if g2 > 0:
                     cand[2] //= g2; cand[3] //= g2
                 
                 if tuple(cand) in current_set:
                     continue
                 if _try_add_point(current_points, cand, D):
                     current_points.append(cand)
                     current_set.add(tuple(cand))
                     added_any = True
                     found = True
                     break
             if not found:
                 break # Stuck

    # Convert to required format
    final_points = []
    for p in current_points:
        x_num, x_den, y_num, y_den = p
        final_points.append([[x_num, x_den], [y_num, y_den]])
        
    return {'D': D, 'points': final_points}

# Fallback to ensure it runs
if __name__ == "__main__":
    construct()
