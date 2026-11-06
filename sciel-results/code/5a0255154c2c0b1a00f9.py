import time
import math
from fractions import Fraction
from itertools import combinations

def is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)

def _is_squarefree(n):
    if n <= 1:
        return True
    for p in range(2, int(math.isqrt(n)) + 1):
        if n % (p * p) == 0:
            return False
    return True

def _check_pairwise_rational(points, D):
    """Check that all pairwise distances are rational (i.e., d^2/D is a perfect square of rationals)."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            # Distance squared in the plane: dx^2 + D*dy^2
            d2 = dx * dx + D * dy * dy
            # d2 must be a perfect square of a rational number
            # d2 = (a/b)^2 where a,b integers
            # So d2 * b^2 = a^2, meaning d2 must be a rational square
            # Since points have rational coords, d2 is rational
            # Check if d2 is a perfect square in Q
            num = d2.numerator
            den = d2.denominator
            if not is_square(num) or not is_square(den):
                return False
    return True

def _no_three_collinear(points):
    """Check no three points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                # Cross product of (p2-p1) and (p3-p1)
                cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                if cross == 0:
                    return False
    return True

def _no_four_concyclic(points, D):
    """Check no four points are concyclic."""
    n = len(points)
    for combo in combinations(range(n), 4):
        i, j, k, l = combo
        p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        # In the (x, y*sqrt(D)) plane, a circle equation is:
        # x^2 + D*y^2 + Ax + By + C = 0
        # For four points to be concyclic, the determinant must be zero:
        # | x1^2 + D*y1^2, x1, y1, 1 |
        # | x2^2 + D*y2^2, x2, y2, 1 | = 0
        # | x3^2 + D*y3^2, x3, y3, 1 |
        # | x4^2 + D*y4^2, x4, y4, 1 |
        
        a1 = x1 * x1 + D * y1 * y1
        a2 = x2 * x2 + D * y2 * y2
        a3 = x3 * x3 + D * y3 * y3
        a4 = x4 * x4 + D * y4 * y4
        
        # Compute 4x4 determinant
        # M = [[a1, x1, y1, 1],
        #      [a2, x2, y2, 1],
        #      [a3, x3, y3, 1],
        #      [a4, x4, y4, 1]]
        
        # Subtract row1 from rows 2,3,4 to simplify
        # R2' = R2 - R1, R3' = R3 - R1, R4' = R4 - R1
        # Then det = det of [[a1,x1,y1,1],[a2-a1,x2-x1,y2-y1,0],[a3-a1,x3-x1,y3-y1,0],[a4-a1,x4-x1,y4-y1,0]]
        
        # Expand along last column:
        # det = 1 * det of 3x3:
        # | a2-a1, x2-x1, y2-y1 |
        # | a3-a1, x3-x1, y3-y1 |
        # | a4-a1, x4-x1, y4-y1 |
        
        b1 = a2 - a1
        b2 = x2 - x1
        b3 = y2 - y1
        c1 = a3 - a1
        c2 = x3 - x1
        c3 = y3 - y1
        d1 = a4 - a1
        d2 = x4 - x1
        d3 = y4 - y1
        
        det3 = (b1 * (c2 * d3 - c3 * d2)
               - b2 * (c1 * d3 - c3 * d1)
               + b3 * (c1 * d2 - c2 * d1))
        
        if det3 == 0:
            return False
    return True

def _validate(points, D):
    """Full validation of a point set."""
    n = len(points)
    # Check distinct
    for i in range(n):
        for j in range(i + 1, n):
            if points[i] == points[j]:
                return False
    return (_check_pairwise_rational(points, D) and
            _no_three_collinear(points) and
            _no_four_concyclic(points, D))

def _try_add_point(base_points, D, max_coord=50, deadline=None):
    """Try to add a 6th point to a 5-point base set."""
    n = len(base_points)
    if n < 3:
        return None
    
    # Generate candidate points in a grid
    candidates = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            candidates.append((Fraction(x), Fraction(y)))
    
    # Filter candidates that maintain rational distances with all base points
    valid_candidates = []
    for cand in candidates:
        if cand in base_points:
            continue
        valid = True
        for bp in base_points:
            dx = cand[0] - bp[0]
            dy = cand[1] - bp[1]
            d2 = dx * dx + D * dy * dy
            if not (is_square(d2.numerator) and is_square(d2.denominator)):
                valid = False
                break
        if valid:
            valid_candidates.append(cand)
    
    # Try each valid candidate as the 6th point
    for cand in valid_candidates:
        if deadline is not None and time.time() > deadline:
            break
        test_set = base_points + [cand]
        if _validate(test_set, D):
            return test_set
    
    return None

def _search_small_grid(D, target_n=6, max_coord=15, deadline=None):
    """Search for a valid set of target_n points in a small grid."""
    points = []
    
    # Start with origin
    points.append((Fraction(0), Fraction(0)))
    
    # Try to build up the set greedily
    for _ in range(target_n - 1):
        if deadline is not None and time.time() > deadline:
            break
        
        best_candidate = None
        for x in range(-max_coord, max_coord + 1):
            for y in range(-max_coord, max_coord + 1):
                if deadline is not None and time.time() > deadline:
                    break
                cand = (Fraction(x), Fraction(y))
                if cand in points:
                    continue
                
                # Check if adding cand maintains all constraints
                test_set = points + [cand]
                
                # Quick check: rational distances
                valid = True
                for bp in points:
                    dx = cand[0] - bp[0]
                    dy = cand[1] - bp[1]
                    d2 = dx * dx + D * dy * dy
                    if not (is_square(d2.numerator) and is_square(d2.denominator)):
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Check collinearity with existing points
                for i in range(len(points)):
                    for j in range(i + 1, len(points)):
                        p1, p2 = points[i], points[j]
                        cross = (p2[0] - p1[0]) * (cand[1] - p1[1]) - (p2[1] - p1[1]) * (cand[0] - p1[0])
                        if cross == 0:
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # Check concyclicity with existing points (need at least 3 existing)
                if len(points) >= 3:
                    for combo in combinations(range(len(points)), 3):
                        i, j, k = combo
                        p1, p2, p3 = points[i], points[j], points[k]
                        x1, y1 = p1
                        x2, y2 = p2
                        x3, y3 = p3
                        x4, y4 = cand
                        
                        a1 = x1 * x1 + D * y1 * y1
                        a2 = x2 * x2 + D * y2 * y2
                        a3 = x3 * x3 + D * y3 * y3
                        a4 = x4 * x4 + D * y4 * y4
                        
                        b1 = a2 - a1
                        b2 = x2 - x1
                        b3 = y2 - y1
                        c1 = a3 - a1
                        c2 = x3 - x1
                        c3 = y3 - y1
                        d1 = a4 - a1
                        d2 = x4 - x1
                        d3 = y4 - y1
                        
                        det3 = (b1 * (c2 * d3 - c3 * d2)
                               - b2 * (c1 * d3 - c3 * d1)
                               + b3 * (c1 * d2 - c2 * d1))
                        
                        if det3 == 0:
                            valid = False
                            break
                    if not valid:
                        continue
                
                if valid:
                    best_candidate = cand
                    break
            if best_candidate is not None:
                break
        
        if best_candidate is None:
            break
        
        points.append(best_candidate)
    
    if len(points) >= target_n and _validate(points, D):
        return points
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    # Known 5-point set (verified)
    base_points_5 = [(Fraction(0), Fraction(0)),
                     (Fraction(-56), Fraction(33)),
                     (Fraction(-16), Fraction(63)),
                     (Fraction(0), Fraction(33)),
                     (Fraction(16), Fraction(63))]
    
    D = 1
    best_points = base_points_5
    
    # Verify the 5-point set
    if not _validate(base_points_5, D):
        # Fallback: use a simpler known valid set
        base_points_5 = [(Fraction(0), Fraction(0)),
                         (Fraction(3), Fraction(4)),
                         (Fraction(0), Fraction(5))]
        if not _validate(base_points_5, D):
            base_points_5 = [(Fraction(0), Fraction(0)),
                             (Fraction(1), Fraction(0)),
                             (Fraction(0), Fraction(1))]
        best_points = base_points_5
    
    # Try to find 6 points using the hexagon-inspired approach
    # Regular hexagon vertices in Q(sqrt(3)) mapped to (x, y*sqrt(3))
    # Hexagon vertices: (1,0), (1/2, sqrt(3)/2), (-1/2, sqrt(3)/2), (-1,0), (-1/2,-sqrt(3)/2), (1/2,-sqrt(3)/2)
    # In (x, y*sqrt(3)) form: (1,0), (1/2, 1/2), (-1/2, 1/2), (-1,0), (-1/2,-1/2), (1/2,-1/2)
    # But these have D=3, not D=1. Let's try D=3.
    
    # Try D=3 with hexagon-inspired points
    D3 = 3
    hex_points = [(Fraction(1), Fraction(0)),
                  (Fraction(1, 2), Fraction(1, 2)),
                  (Fraction(-1, 2), Fraction(1, 2)),
                  (Fraction(-1), Fraction(0)),
                  (Fraction(-1, 2), Fraction(-1, 2)),
                  (Fraction(1, 2), Fraction(-1, 2))]
    
    if _validate(hex_points, D3):
        best_points = hex_points
        D = D3
    else:
        # Try to find 6 points with D=1
        result = _try_add_point(base_points_5, D, max_coord=30, deadline=deadline)
        if result is not None:
            best_points = result
        else:
            # Try grid search
            result = _search_small_grid(D, target_n=6, max_coord=10, deadline=deadline)
            if result is not None:
                best_points = result
    
    # Final check
    if not _validate(best_points, D):
        # Fallback to the 5-point set
        best_points = base_points_5
        D = 1
        if not _validate(best_points, D):
            # Minimal fallback
            best_points = [(Fraction(0), Fraction(0)),
                           (Fraction(1), Fraction(0)),
                           (Fraction(0), Fraction(1))]
    
    return {'D': D, 'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in best_points]}
