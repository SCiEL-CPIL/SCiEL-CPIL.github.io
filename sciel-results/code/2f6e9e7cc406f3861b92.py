"""Program for constructing integral point sets in general position."""
import random
from fractions import Fraction
import math

# EVOLVE-BLOCK-START

def is_squarefree(n):
    """Check if n is squarefree."""
    if n < 0:
        n = -n
    if n == 0:
        return False
    i = 2
    while i * i <= n:
        if n % (i * i) == 0:
            return False
        i += 1
    return True

def is_perfect_square_frac(frac):
    """Check if a Fraction is a perfect square of a rational number."""
    if frac < 0:
        return False
    if frac == 0:
        return True
    num = frac.numerator
    den = frac.denominator
    sqrt_num = int(math.isqrt(num))
    sqrt_den = int(math.isqrt(den))
    if sqrt_num * sqrt_num != num:
        return False
    if sqrt_den * sqrt_den != den:
        return False
    return True

def rational_dist_sq(p1, p2, D):
    """Compute squared distance between two points (x1, y1*sqrt(D)) and (x2, y2*sqrt(D))."""
    x1 = Fraction(p1[0], p1[1])
    y1 = Fraction(p1[2], p1[3])
    x2 = Fraction(p2[0], p2[1])
    y2 = Fraction(p2[2], p2[3])
    dx = x1 - x2
    dy = y1 - y2
    return dx * dx + D * dy * dy

def all_pairwise_rational(points, D):
    """Check if all pairwise distances are rational."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            d2 = rational_dist_sq(points[i], points[j], D)
            if not is_perfect_square_frac(d2):
                return False
    return True

def no_three_collinear(points):
    """Check no three points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1 = Fraction(points[i][0], points[i][1])
                y1 = Fraction(points[i][2], points[i][3])
                x2 = Fraction(points[j][0], points[j][1])
                y2 = Fraction(points[j][2], points[j][3])
                x3 = Fraction(points[k][0], points[k][1])
                y3 = Fraction(points[k][2], points[k][3])
                if (x2 - x1) * (y3 - y1) == (x3 - x1) * (y2 - y1):
                    return False
    return True

def _four_concyclic_check(p1, p2, p3, p4, D):
    """Check if 4 points are concyclic using determinant."""
    x1 = Fraction(p1[0], p1[1])
    y1 = Fraction(p1[2], p1[3])
    x2 = Fraction(p2[0], p2[1])
    y2 = Fraction(p2[2], p2[3])
    x3 = Fraction(p3[0], p3[1])
    y3 = Fraction(p3[2], p3[3])
    x4 = Fraction(p4[0], p4[1])
    y4 = Fraction(p4[2], p4[3])
    
    u1 = x1 * x1 + D * y1 * y1
    u2 = x2 * x2 + D * y2 * y2
    u3 = x3 * x3 + D * y3 * y3
    u4 = x4 * x4 + D * y4 * y4
    
    # Determinant of [[u_i, x_i, y_i, 1]]
    a11, a12, a13, a14 = u1, x1, y1, Fraction(1)
    a21, a22, a23, a24 = u2, x2, y2, Fraction(1)
    a31, a32, a33, a34 = u3, x3, y3, Fraction(1)
    a41, a42, a43, a44 = u4, x4, y4, Fraction(1)
    
    det = (a11 * (a22 * a33 * a44 - a22 * a34 * a43 - a23 * a32 * a44 + a23 * a34 * a42 + a24 * a32 * a43 - a24 * a33 * a42)
         - a12 * (a21 * a33 * a44 - a21 * a34 * a43 - a23 * a31 * a44 + a23 * a34 * a41 + a24 * a31 * a43 - a24 * a33 * a41)
         + a13 * (a21 * a32 * a44 - a21 * a34 * a42 - a22 * a31 * a44 + a22 * a34 * a41 + a24 * a31 * a42 - a24 * a32 * a41)
         - a14 * (a21 * a32 * a43 - a21 * a33 * a42 - a22 * a31 * a43 + a22 * a33 * a41 + a23 * a31 * a42 - a23 * a32 * a41))
    
    return det == 0

def no_four_concyclic(points, D):
    """Check no four points are concyclic."""
    n = len(points)
    if n < 4:
        return True
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if _four_concyclic_check(points[i], points[j], points[k], points[l], D):
                        return False
    return True

def validate(points, D):
    """Full validation of the point set."""
    if not is_squarefree(D):
        return False
    if D < 1 or D > 10**12:
        return False
    n = len(points)
    if n < 3 or n > 12:
        return False
    for p in points:
        for val in [p[0], p[1], p[2], p[3]]:
            if abs(val) > 10**40:
                return False
        if p[1] == 0 or p[3] == 0:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            if points[i] == points[j]:
                return False
    if not all_pairwise_rational(points, D):
        return False
    if not no_three_collinear(points):
        return False
    if not no_four_concyclic(points, D):
        return False
    return True

def circle_point(D, m, n_val):
    """Generate a point on the unit circle in the (x, y*sqrt(D)) plane.
    x = 2m/(m^2 + D*n^2), y = (m^2 - D*n^2)/(m^2 + D*n^2)
    Distance from origin: sqrt(x^2 + D*y^2) = 1 (rational)
    """
    denom = m * m + D * n_val * n_val
    if denom == 0:
        return None
    x_num = 2 * m
    x_den = denom
    y_num = m * m - D * n_val * n_val
    y_den = denom
    # Reduce fractions
    from math import gcd
    g1 = gcd(abs(x_num), x_den) if x_num != 0 else x_den
    g2 = gcd(abs(y_num), y_den) if y_num != 0 else y_den
    x_num //= g1
    x_den //= g1
    y_num //= g2
    y_den //= g2
    return (x_num, x_den, y_num, y_den)

def generate_off_circle_point(D, existing_points, max_coord=200):
    """Try to find a point with rational distances to all existing points."""
    for _ in range(5000):
        x = random.randint(-max_coord, max_coord)
        y = random.randint(-max_coord, max_coord)
        if x == 0 and y == 0:
            continue
        p = (x, 1, y, 1)
        valid = True
        for q in existing_points:
            d2 = rational_dist_sq(p, q, D)
            if not is_perfect_square_frac(d2):
                valid = False
                break
        if not valid:
            continue
        # Check no three collinear
        collinear = False
        for i in range(len(existing_points)):
            for j in range(i + 1, len(existing_points)):
                x1 = Fraction(p[0], p[1])
                y1 = Fraction(p[2], p[3])
                x2 = Fraction(existing_points[i][0], existing_points[i][1])
                y2 = Fraction(existing_points[i][2], existing_points[i][3])
                x3 = Fraction(existing_points[j][0], existing_points[j][1])
                y3 = Fraction(existing_points[j][2], existing_points[j][3])
                if (x2 - x1) * (y3 - y1) == (x3 - x1) * (y2 - y1):
                    collinear = True
                    break
            if collinear:
                break
        if collinear:
            continue
        # Check no four concyclic
        concyclic = False
        for i in range(len(existing_points)):
            for j in range(i + 1, len(existing_points)):
                for k in range(j + 1, len(existing_points)):
                    if _four_concyclic_check(p, existing_points[i], existing_points[j], existing_points[k], D):
                        concyclic = True
                        break
                if concyclic:
                    break
            if concyclic:
                break
        if not concyclic:
            return p
    return None

def construct():
    """Construct a set of integral points in general position."""
    D = 2
    
    # Generate points on the unit circle using Pythagorean parameterization
    circle_points = []
    # Use various (m, n) pairs to get distinct points
    params = [(1, 0), (0, 1), (1, 1), (2, 1), (3, 1), (1, 2), (3, 2), (4, 1), (5, 1), (2, 3)]
    seen = set()
    for m, n_val in params:
        p = circle_point(D, m, n_val)
        if p is None:
            continue
        if p in seen:
            continue
        # Check rational distance to all existing circle points
        valid = True
        for q in circle_points:
            d2 = rational_dist_sq(p, q, D)
            if not is_perfect_square_frac(d2):
                valid = False
                break
        if valid:
            # Check no three collinear
            collinear = False
            for i in range(len(circle_points)):
                for j in range(i + 1, len(circle_points)):
                    x1 = Fraction(p[0], p[1])
                    y1 = Fraction(p[2], p[3])
                    x2 = Fraction(circle_points[i][0], circle_points[i][1])
                    y2 = Fraction(circle_points[i][2], circle_points[i][3])
                    x3 = Fraction(circle_points[j][0], circle_points[j][1])
                    y3 = Fraction(circle_points[j][2], circle_points[j][3])
                    if (x2 - x1) * (y3 - y1) == (x3 - x1) * (y2 - y1):
                        collinear = True
                        break
                if collinear:
                    break
            if not collinear:
                circle_points.append(p)
                seen.add(p)
        if len(circle_points) >= 5:
            break
    
    # We have points on a circle. Now try to add points off the circle.
    # First, let's try to get at least 4 valid points (no four concyclic means we can have at most 3 on any circle)
    # Wait - if all points are on the same circle, any 4 are concyclic!
    # So we need at most 3 points on any given circle.
    # Strategy: use 3 points on the unit circle, then add more points off the circle.
    
    # Let's use 3 points on the unit circle
    base_points = circle_points[:3]
    
    # Try to extend with off-circle points
    for attempt in range(50):
        points = list(base_points)
        random.shuffle(points)
        # Reorder to keep first 3
        points = base_points + []
        
        target = 6
        for _ in range(target - len(points)):
            p = generate_off_circle_point(D, points, max_coord=100)
            if p is None:
                break
            points.append(p)
        
        if len(points) >= 5 and validate(points, D):
            return {'D': D, 'points': [[[p[0], p[1]], [p[2], p[3]]] for p in points]}
    
    # Fallback: try with D=1 and known configurations
    # 4 points: use a known set
    test_sets = [
        (1, [(0,1,0,1), (3,1,4,1), (0,1,5,1), (4,1,3,1)]),
        (2, [(0,1,0,1), (2,1,1,1), (1,1,1,1), (1,1,-1,1)]),
    ]
    
    for D_test, pts in test_sets:
        if validate(pts, D_test):
            return {'D': D_test, 'points': [[[p[0], p[1]], [p[2], p[3]]] for p in pts]}
    
    # Ultimate fallback: 3 points
    pts = [(0,1,0,1), (1,1,0,1), (0,1,1,1)]
    D_test = 2
    if validate(pts, D_test):
        return {'D': D_test, 'points': [[[p[0], p[1]], [p[2], p[3]]] for p in pts]}
    
    return {'D': 1, 'points': [[[0,1],[0,1]], [[1,1],[0,1]], [[0,1],[1,1]]]}

# EVOLVE-BLOCK-END
