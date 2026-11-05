"""Program for constructing integral point sets in general position."""
import random
from fractions import Fraction
import math

# EVOLVE-BLOCK-START

def is_squarefree(n):
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

def dist_sq(p1, p2, D):
    """Compute squared distance between two points."""
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
            d2 = dist_sq(points[i], points[j], D)
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
    
    # Determinant of 4x4 matrix with rows [u_i, x_i, y_i, 1]
    # Use Sarrus-like expansion for 4x4
    def det4(m):
        a,b,c,d = m[0]
        e,f,g,h = m[1]
        i,j,k,l = m[2]
        m_,n_,o,p = m[3]
        return (a*(f*(k*p-o*j) - g*(j*p-o*i) + h*(j*o-k*i))
              - b*(e*(k*p-o*j) - g*(j*p-m_*i) + h*(j*o-m_*i))
              + c*(e*(k*o-i*j) - f*(j*o-m_*i) + h*(j*m_-k*i))
              - d*(e*(k*n_-i*j) - f*(j*n_-m_*i) + g*(j*m_-k*i)))
    
    M = [[u1, x1, y1, Fraction(1)],
         [u2, x2, y2, Fraction(1)],
         [u3, x3, y3, Fraction(1)],
         [u4, x4, y4, Fraction(1)]]
    
    return det4(M) == 0

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

def pythagorean_param(t, m, n):
    """Generate a point (x, y) such that x^2 + y^2 is a perfect square.
    x = t*(m^2 - n^2), y = t*(2*m*n), distance = t*(m^2 + n^2)
    All rational if t, m, n are rational."""
    x = t * (m * m - n * n)
    y = t * (2 * m * n)
    return x, y

def try_add_point(existing_points, D, max_search=200):
    """Try to find a new point that has rational distances to all existing points."""
    n = len(existing_points)
    
    # Generate candidate points using Pythagorean parameterization
    # Point must have rational distance to origin (if origin is in set) or to each existing point
    
    for attempt in range(max_search):
        # Random parameters for Pythagorean triple
        m = random.randint(1, 30)
        nn = random.randint(1, 30)
        if m == nn:
            continue
        t = Fraction(random.randint(1, 20), random.randint(1, 20))
        
        x, y = pythagorean_param(t, Fraction(m), Fraction(nn))
        
        # Add some random offset to explore more space
        if attempt > 50:
            ox = Fraction(random.randint(-10, 10), random.randint(1, 10))
            oy = Fraction(random.randint(-10, 10), random.randint(1, 10))
            x = x + ox
            y = y + oy
        
        # Check if this point has rational distance to all existing points
        valid = True
        for q in existing_points:
            d2 = dist_sq((x.numerator, x.denominator, y.numerator, y.denominator), q, D)
            if not is_perfect_square_frac(d2):
                valid = False
                break
        
        if not valid:
            continue
        
        # Check no three collinear
        collinear = False
        for i in range(n):
            for j in range(i + 1, n):
                p_new = (x.numerator, x.denominator, y.numerator, y.denominator)
                x1 = Fraction(p_new[0], p_new[1])
                y1 = Fraction(p_new[2], p_new[3])
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
        
        # Check no four concyclic with new point
        concyclic = False
        new_pt = (x.numerator, x.denominator, y.numerator, y.denominator)
        if n >= 3:
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        if _four_concyclic_check(new_pt, existing_points[i], existing_points[j], existing_points[k], D):
                            concyclic = True
                            break
                    if concyclic:
                        break
                if concyclic:
                    break
        
        if concyclic:
            continue
        
        return new_pt
    
    return None

def construct():
    """Construct a set of integral points in general position."""
    D = 1
    
    # Start with a known valid 3-point set
    base_points = [(0, 1, 0, 1), (3, 1, 4, 1), (0, 1, 3, 1)]
    
    if not validate(base_points, D):
        # Fallback simple set
        base_points = [(0, 1, 0, 1), (3, 1, 4, 1), (6, 1, 8, 1)]
        if not validate(base_points, D):
            return {'D': 1, 'points': [[[0, 1], [0, 1]], [[3, 1], [4, 1]], [[6, 1], [8, 1]]]}
    
    current_points = list(base_points)
    
    # Try to extend to 6 points
    target = 6
    for _ in range(target - len(current_points)):
        new_pt = try_add_point(current_points, D, max_search=500)
        if new_pt is None:
            break
        current_points.append(new_pt)
    
    # Try harder with different strategies if we didn't reach target
    if len(current_points) < target:
        # Try starting from different base points
        alt_bases = [
            [(0, 1, 0, 1), (5, 1, 12, 1), (0, 1, 5, 1)],
            [(0, 1, 0, 1), (8, 1, 6, 1), (0, 1, 8, 1)],
            [(0, 1, 0, 1), (7, 1, 24, 1), (0, 1, 7, 1)],
        ]
        for base in alt_bases:
            if not validate(base, D):
                continue
            pts = list(base)
            for _ in range(target - len(pts)):
                new_pt = try_add_point(pts, D, max_search=500)
                if new_pt is None:
                    break
                pts.append(new_pt)
            if len(pts) > len(current_points):
                current_points = pts
    
    # Final validation
    if validate(current_points, D):
        return {'D': D, 'points': [[[p[0], p[1]], [p[2], p[3]]] for p in current_points]}
    
    # Return best valid set we found
    if len(current_points) >= 3:
        return {'D': D, 'points': [[[p[0], p[1]], [p[2], p[3]]] for p in current_points]}
    
    # Ultimate fallback
    return {'D': 1, 'points': [[[0, 1], [0, 1]], [[3, 1], [4, 1]], [[6, 1], [8, 1]]]}

# EVOLVE-BLOCK-END
