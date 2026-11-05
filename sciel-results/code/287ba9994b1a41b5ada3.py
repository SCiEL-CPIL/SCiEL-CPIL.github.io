"""Rewritten program for integral point sets in general position.
Searches for larger configurations than the 5-point seed.
"""
import time
import random
from math import isqrt, gcd
from fractions import Fraction
from itertools import combinations

# Deadline guard
T0 = time.time()
TIME_LIMIT = 115  # seconds, safety margin before 120s

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _sqdist(p, q, D):
    """Squared distance between p=(x1,y1) and q=(x2,y2) in the metric
    where actual point is (x, y*sqrt(D)).
    Returns Fraction."""
    x1, y1 = p
    x2, y2 = q
    dx = x1 - x2
    dy = y1 - y2
    return dx * dx + D * dy * dy

def _check_rational_distances(points, D):
    """Check all pairwise squared distances are perfect squares of rationals."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            sd = _sqdist(points[i], points[j], D)
            if sd < 0:
                return False
            # sd must be a perfect square in Q
            num, den = sd.numerator, sd.denominator
            if not _is_square(num) or not _is_square(den):
                return False
    return True

def _check_no_three_collinear(points, D):
    """Check no three points are collinear using exact arithmetic."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                # Cross product: (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
                # In our metric, y is scaled by sqrt(D), so collinearity in (x, y*sqrt(D))
                # means (x2-x1)*(y3-y1)*sqrt(D) - (x3-x1)*(y2-y1)*sqrt(D) = 0
                # which simplifies to (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1) = 0
                cross = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
                if cross == 0:
                    return False
    return True

def _check_no_four_concyclic(points, D):
    """Check no four points are concyclic.
    Four points (x1,y1),(x2,y2),(x3,y3),(x4,y4) in metric (x, y*sqrt(D)) are concyclic
    iff the determinant of the circle equation vanishes.
    
    A circle in (X, Y) space (where X=x, Y=y*sqrt(D)) has equation:
    X^2 + Y^2 + aX + bY + c = 0
    
    For point i: xi^2 + Di*yi^2 + a*xi + b*sqrt(D)*yi + c = 0
    
    This is linear in a, b*sqrt(D), c. For 4 points to be concyclic, the 4x4 system
    must be singular. We can check via the determinant of:
    | xi^2 + D*yi^2  xi  sqrt(D)*yi  1 | for i=1..4
    
    But sqrt(D)*yi is irrational if D is not a square. So we need to handle this carefully.
    
    Alternative: use the condition that 4 points are concyclic iff the cross ratio of 
    their distances is real, or use the determinant condition in a rational form.
    
    Actually, let's use the standard circle determinant. For points (Xi, Yi) where Xi=xi, Yi=yi*sqrt(D):
    The determinant:
    | Xi^2+Yi^2  Xi  Yi  1 |
    must be zero for all 4 points.
    
    Since Yi = yi*sqrt(D), the entries Xi^2+Yi^2 = xi^2 + D*yi^2 (rational), Xi=xi (rational), Yi=yi*sqrt(D) (irrational if D not square).
    
    For the determinant to be zero, we can separate rational and irrational parts.
    Let's write the 4x4 matrix M with rows for each point.
    M[i] = [xi^2 + D*yi^2, xi, yi*sqrt(D), 1]
    
    det(M) = 0. Since sqrt(D) is irrational (for D not a perfect square), det(M) is of the form A + B*sqrt(D) where A, B are rational. For det(M)=0, we need A=0 and B=0.
    
    This is complex. Let me use a different approach: the condition for 4 points to be concyclic can be checked by computing the circle through 3 points and checking if the 4th lies on it.
    
    For 3 points, the circle is uniquely determined (if not collinear). We can compute the circle equation coefficients and check the 4th point.
    """
    n = len(points)
    for quad in combinations(range(n), 4):
        i, j, k, l = quad
        p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        # Circle through p1, p2, p3 in (X, Y) = (x, y*sqrt(D)) space.
        # Equation: X^2 + Y^2 + aX + bY + c = 0
        # For point 1: x1^2 + D*y1^2 + a*x1 + b*sqrt(D)*y1 + c = 0
        # Let A = a, B = b*sqrt(D), C = c. Then:
        # x1^2 + D*y1^2 + A*x1 + B*y1 + C = 0  ... (1)
        # x2^2 + D*y2^2 + A*x2 + B*y2 + C = 0  ... (2)
        # x3^2 + D*y3^2 + A*x3 + B*y3 + C = 0  ... (3)
        
        # Subtract (1) from (2) and (3):
        # (x2^2-x1^2) + D*(y2^2-y1^2) + A*(x2-x1) + B*(y2-y1) = 0
        # (x3^2-x1^2) + D*(y3^2-y1^2) + A*(x3-x1) + B*(y3-y1) = 0
        
        # This is a 2x2 linear system in A, B.
        a11 = x2 - x1
        a12 = y2 - y1
        a21 = x3 - x1
        a22 = y3 - y1
        b1 = -((x2*x2 - x1*x1) + D*(y2*y2 - y1*y1))
        b2 = -((x3*x3 - x1*x1) + D*(y3*y3 - y1*y1))
        
        det = a11 * a22 - a12 * a21
        if det == 0:
            # Points are collinear, skip (already checked)
            continue
        
        A = (b1 * a22 - b2 * a12) / det
        B = (a11 * b2 - a21 * b1) / det
        
        # C = -(x1^2 + D*y1^2 + A*x1 + B*y1)
        C = -(x1*x1 + D*y1*y1 + A*x1 + B*y1)
        
        # Check if p4 lies on the circle:
        # x4^2 + D*y4^2 + A*x4 + B*y4 + C == 0
        val = x4*x4 + D*y4*y4 + A*x4 + B*y4 + C
        if val == 0:
            return False
    return True

def _is_valid(points, D):
    """Full validation."""
    if not _check_rational_distances(points, D):
        return False
    if not _check_no_three_collinear(points, D):
        return False
    if not _check_no_four_concyclic(points, D):
        return False
    return True

def _format_result(points, D):
    """Convert points list of Fraction tuples to the required format."""
    result_points = []
    for x, y in points:
        result_points.append([
            [x.numerator, x.denominator],
            [y.numerator, y.denominator]
        ])
    return {'D': D, 'points': result_points}

def _search_D1():
    """Search for configurations with D=1 (integer points with integer distances)."""
    best = None
    best_n = 0
    
    # Start with known 5-point configuration
    base = [(Fraction(0), Fraction(0)), 
            (Fraction(-56), Fraction(33)),
            (Fraction(-16), Fraction(63)),
            (Fraction(0), Fraction(33)),
            (Fraction(16), Fraction(63))]
    
    if _is_valid(base, 1):
        best = base
        best_n = 5
    
    # Try to add points to the base configuration
    # Use a local search / hill climbing approach
    for target_n in range(6, 13):
        if time.time() - T0 > TIME_LIMIT - 10:
            break
        found = False
        # Try adding random points
        for attempt in range(50000):
            if time.time() - T0 > TIME_LIMIT - 5:
                break
            # Generate a random candidate point
            x = Fraction(random.randint(-200, 200), random.randint(1, 200))
            y = Fraction(random.randint(-200, 200), random.randint(1, 200))
            
            # Simplify
            x = x / gcd(x.numerator, x.denominator) if x.numerator != 0 else Fraction(0)
            y = y / gcd(y.numerator, y.denominator) if y.numerator != 0 else Fraction(0)
            
            # Check bounds
            if abs(x.numerator) > 10**40 or x.denominator > 10**40:
                continue
            if abs(y.numerator) > 10**40 or y.denominator > 10**40:
                continue
            
            # Check if this point can be added to the current best
            candidate = best + [(x, y)]
            if len(candidate) != len(set(candidate)):
                continue
            
            if _check_rational_distances(candidate, 1) and \
               _check_no_three_collinear(candidate, 1) and \
               _check_no_four_concyclic(candidate, 1):
                best = candidate
                best_n = target_n
                found = True
                break
        
        if not found and time.time() - T0 > TIME_LIMIT - 10:
            break
    
    return best, best_n

def _search_general_D():
    """Search for configurations with D > 1."""
    best = None
    best_n = 0
    best_D = 1
    
    # Try small squarefree D values
    for D in [2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 46, 47, 51, 53, 55, 57, 58, 59, 61, 62, 65, 66, 67, 69, 70, 71, 73, 74, 77, 78, 79, 82, 83, 85, 86, 87, 89, 90, 93, 94, 95, 97]:
        if time.time() - T0 > TIME_LIMIT - 10:
            break
        
        # Try small integer/rational points
        for attempt in range(20000):
            if time.time() - T0 > TIME_LIMIT - 5:
                break
            
            # Generate a few random points
            n_pts = random.randint(3, 5)
            pts = []
            valid = True
            for _ in range(n_pts):
                x = Fraction(random.randint(-50, 50), random.randint(1, 50))
                y = Fraction(random.randint(-50, 50), random.randint(1, 50))
                pts.append((x, y))
            
            # Remove duplicates
            if len(pts) != len(set(pts)):
                continue
            
            if _is_valid(pts, D):
                if n_pts > best_n:
                    best = pts
                    best_n = n_pts
                    best_D = D
    
    return best, best_n, best_D

def construct():
    """Main entry point. Searches for the largest valid configuration."""
    global T0
    T0 = time.time()
    
    best_points = None
    best_n = 0
    best_D = 1
    
    # First, verify the seed
    seed_points = [(Fraction(0), Fraction(0)), 
                   (Fraction(-56), Fraction(33)),
                   (Fraction(-16), Fraction(63)),
                   (Fraction(0), Fraction(33)),
                   (Fraction(16), Fraction(63))]
    
    if _is_valid(seed_points, 1):
        best_points = seed_points
        best_n = 5
        best_D = 1
    
    # Try to extend with D=1
    if time.time() - T0 < TIME_LIMIT - 20:
        pts_d1, n_d1 = _search_D1()
        if pts_d1 and n_d1 > best_n:
            best_points = pts_d1
            best_n = n_d1
            best_D = 1
    
    # Try general D
    if time.time() - T0 < TIME_LIMIT - 10:
        pts_gen, n_gen, D_gen = _search_general_D()
        if pts_gen and n_gen > best_n:
            best_points = pts_gen
            best_n = n_gen
            best_D = D_gen
    
    # Ensure we always return at least the seed
    if best_points is None:
        best_points = seed_points
        best_D = 1
    
    return _format_result(best_points, best_D)
