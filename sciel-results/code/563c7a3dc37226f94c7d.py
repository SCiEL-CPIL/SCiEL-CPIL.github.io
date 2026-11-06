"""Rewritten program: incremental search for integral point sets with exact rational arithmetic."""
from fractions import Fraction
from math import isqrt, gcd
from itertools import combinations
import time
import random

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _dist_sq(p1, p2, D):
    """Squared distance between two points (x1,y1*sqrt(D)) and (x2,y2*sqrt(D)), as Fraction."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + D * dy * dy

def _check_rational_distances(points, D):
    """Check all pairwise squared distances are perfect squares of rationals."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            dsq = _dist_sq(points[i], points[j], D)
            if dsq <= 0:
                return False
            # dsq must be a perfect square of a rational
            num, den = dsq.numerator, dsq.denominator
            if not _is_square(num) or not _is_square(den):
                return False
    return True

def _check_no_three_collinear(points):
    """Check no three points are collinear using exact arithmetic."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                # Cross product (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
                cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                if cross == 0:
                    return False
    return True

def _check_no_four_concyclic(points, D):
    """Check no four points are concyclic using exact arithmetic."""
    n = len(points)
    # For four points to be concyclic, the determinant of the circle equation must be zero.
    # Circle: x^2 + y^2*D + Dx + Ey + F = 0
    # For point (x,y): x^2 + y^2*D + Dx + Ey + F = 0
    # Four points concyclic iff the 4x4 determinant is zero.
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for m in range(k + 1, n):
                    pts = [points[i], points[j], points[k], points[m]]
                    # Build 4x4 matrix: [x^2 + y^2*D, x, y, 1]
                    det = 0
                    # Compute determinant of 4x4 matrix
                    M = []
                    for (x, y) in pts:
                        val = x * x + D * y * y
                        M.append([val, x, y, Fraction(1)])
                    # Gaussian elimination for 4x4
                    det_val = _det4(M)
                    if det_val == 0:
                        return False
    return True

def _det4(M):
    """Determinant of 4x4 matrix with Fraction entries."""
    n = 4
    sign = Fraction(1)
    det = Fraction(0)
    # Use cofactor expansion along first row
    for j in range(n):
        minor = []
        for i in range(1, n):
            row = []
            for k in range(n):
                if k != j:
                    row.append(M[i][k])
            minor.append(row)
        det += sign * M[0][j] * _det3(minor)
        sign = -sign
    return det

def _det3(M):
    """Determinant of 3x3 matrix with Fraction entries."""
    return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
            - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
            + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))

def _check_valid(points, D):
    """Full validity check."""
    if len(points) < 3:
        return False
    # Check distinctness
    seen = set()
    for p in points:
        key = (p[0], p[1])
        if key in seen:
            return False
        seen.add(key)
    if not _check_rational_distances(points, D):
        return False
    if not _check_no_three_collinear(points):
        return False
    if not _check_no_four_concyclic(points, D):
        return False
    return True

def _try_add_point(points, D, x, y):
    """Try to add point (x, y) to the set. Return True if valid."""
    new_points = points + [[Fraction(x), Fraction(y)]]
    return _check_valid(new_points, D)

def _search_incremental(D, max_points, time_limit, rng):
    """Incrementally build a valid point set by adding one point at a time."""
    start_time = time.time()
    
    # Start with a small valid seed
    # Try to find initial 3-point set
    best = None
    
    # Use known small configurations as seeds
    seeds = [
        [[Fraction(0), Fraction(0)], [Fraction(3), Fraction(4)], [Fraction(0), Fraction(5)]],  # 3-4-5 triangle
        [[Fraction(0), Fraction(0)], [Fraction(5), Fraction(0)], [Fraction(0), Fraction(12)]],  # 5-12-13
        [[Fraction(0), Fraction(0)], [Fraction(8), Fraction(0)], [Fraction(0), Fraction(6)]],   # 6-8-10
    ]
    
    # Check seeds are valid
    valid_seeds = []
    for s in seeds:
        if _check_valid(s, D):
            valid_seeds.append(s)
    
    if not valid_seeds:
        return None
    
    # Try each seed
    for seed in valid_seeds:
        current = list(seed)
        if best is None or len(current) > len(best):
            best = list(current)
        
        # Try to extend
        for _ in range(50):  # Try many extensions
            if time.time() - start_time > time_limit - 5:
                break
            if len(current) >= max_points:
                break
            
            # Try random candidate points
            found = False
            for _ in range(200):
                if time.time() - start_time > time_limit - 5:
                    break
                # Random integer point in a bounded region
                x = rng.randint(-50, 50)
                y = rng.randint(-50, 50)
                if _try_add_point(current, D, x, y):
                    current.append([Fraction(x), Fraction(y)])
                    found = True
                    if best is None or len(current) > len(best):
                        best = list(current)
                    break
            if not found:
                # Try a few more with different strategy
                for _ in range(100):
                    if time.time() - start_time > time_limit - 5:
                        break
                    x = rng.randint(-100, 100)
                    y = rng.randint(-100, 100)
                    if _try_add_point(current, D, x, y):
                        current.append([Fraction(x), Fraction(y)])
                        found = True
                        if best is None or len(current) > len(best):
                            best = list(current)
                        break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    rng = random.Random(42)
    
    best_points = None
    best_D = 1
    
    # Try D=1 first (integer points)
    for max_pts in range(6, 12):
        if time.time() > deadline - 30:
            break
        result = _search_incremental(1, max_pts, deadline - start_time, rng)
        if result and (best_points is None or len(result) > len(best_points)):
            best_points = result
            best_D = 1
    
    # If we found at least 5, try to improve
    if best_points is None:
        # Fallback: use the known 5-point set
        points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        best_points = [[Fraction(x), Fraction(y)] for x, y in points]
        best_D = 1
    elif len(best_points) <= 5:
        # Try a more targeted search for 6 points
        if time.time() < deadline - 20:
            # Try specific known 6-point configurations or search harder
            result = _search_incremental(1, 7, deadline - start_time, rng)
            if result and len(result) > len(best_points):
                best_points = result
                best_D = 1
    
    # Convert to required format
    out_points = []
    for p in best_points:
        x_num = p[0].numerator
        x_den = p[0].denominator
        y_num = p[1].numerator
        y_den = p[1].denominator
        out_points.append([[x_num, x_den], [y_num, y_den]])
    
    return {'D': best_D, 'points': out_points}
