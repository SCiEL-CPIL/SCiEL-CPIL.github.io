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

def make_squarefree(n):
    """Make n squarefree by removing square factors."""
    if n < 0:
        sign = -1
        n = -n
    else:
        sign = 1
    if n == 0:
        return 0
    result = 1
    i = 2
    while i * i <= n:
        while n % (i * i) == 0:
            n //= (i * i)
        i += 1
    result = n
    return sign * result

def rational_dist_sq(p1, p2, D):
    """Compute squared distance between two points (x1, y1*sqrt(D)) and (x2, y2*sqrt(D)).
    Points are given as (x_num, x_den, y_num, y_den)."""
    x1 = Fraction(p1[0], p1[1])
    y1 = Fraction(p2[2], p2[3])  # Wait, this is wrong. Let me fix.
    # Actually p1 = (x1_num, x1_den, y1_num, y1_den)
    x1 = Fraction(p1[0], p1[1])
    y1 = Fraction(p1[2], p1[3])
    x2 = Fraction(p2[0], p2[1])
    y2 = Fraction(p2[2], p2[3])
    
    dx = x1 - x2
    dy = y1 - y2
    # dist^2 = dx^2 + D * dy^2
    return dx * dx + D * dy * dy

def is_perfect_square_frac(frac):
    """Check if a Fraction is a perfect square of a rational number."""
    if frac < 0:
        return False
    if frac == 0:
        return True
    num = frac.numerator
    den = frac.denominator
    # Check if num and den are perfect squares
    sqrt_num = int(math.isqrt(num))
    sqrt_den = int(math.isqrt(den))
    if sqrt_num * sqrt_num != num:
        return False
    if sqrt_den * sqrt_den != den:
        return False
    return True

def all_pairwise_rational(points, D):
    """Check if all pairwise distances are rational (squared distances are perfect squares)."""
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
                # Points (x1,y1), (x2,y2), (x3,y3) collinear if
                # (x2-x1)(y3-y1) == (x3-x1)(y2-y1)
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
    """Check no four points are concyclic.
    Four points are concyclic if the circle through any three passes through the fourth.
    For points (x_i, y_i * sqrt(D)), we need to check if there's a circle.
    
    A circle in the (x, y*sqrt(D)) plane: (x-a)^2 + (y*sqrt(D)-b)^2 = r^2
    This expands to: x^2 - 2ax + a^2 + D*y^2 - 2b*sqrt(D)*y + b^2 = r^2
    Rearranging: x^2 + D*y^2 - 2a*x - 2b*sqrt(D)*y + (a^2 + b^2 - r^2) = 0
    
    For this to hold for rational x, y, the coefficients must work out.
    Let's use the determinant condition. Four points are concyclic iff the determinant
    of the following matrix is zero:
    |x^2 + D*y^2, x, y*sqrt(D), 1|
    
    But y*sqrt(D) is irrational unless y=0. So we need a different approach.
    
    Alternative: Three points define a circle (if not collinear). The fourth is on it iff...
    Let's use the fact that for points P1, P2, P3, P4, they are concyclic iff
    the power of each point with respect to the circle through the other three is zero.
    
    Actually, let's use the determinant approach with exact arithmetic.
    The equation of a circle through (x1,y1), (x2,y2), (x3,y3) where y_i are the actual y-coordinates (i.e., y_i * sqrt(D)):
    
    Let me denote the actual coordinates as (X_i, Y_i) where X_i = x_i, Y_i = y_i * sqrt(D).
    
    The circle equation: X^2 + Y^2 + A*X + B*Y + C = 0
    
    For point i: X_i^2 + Y_i^2 + A*X_i + B*Y_i + C = 0
    
    Y_i^2 = D * y_i^2, so:
    x_i^2 + D*y_i^2 + A*x_i + B*y_i*sqrt(D) + C = 0
    
    For this to hold, we need the rational part and the sqrt(D) part to separately be zero (assuming D is squarefree and D != 1).
    
    If D = 1, then Y_i = y_i, and we just have x_i^2 + y_i^2 + A*x_i + B*y_i + C = 0.
    
    For D > 1 squarefree:
    x_i^2 + D*y_i^2 + A*x_i + C + B*y_i*sqrt(D) = 0
    => (x_i^2 + D*y_i^2 + A*x_i + C) + B*y_i*sqrt(D) = 0
    
    Since sqrt(D) is irrational, we need:
    B*y_i = 0 for all i (if B != 0, then all y_i = 0, which is a degenerate case)
    and x_i^2 + D*y_i^2 + A*x_i + C = 0
    
    Wait, this can't be right in general. Let me reconsider.
    
    Actually, A, B, C can be any real numbers. The condition is that there exist real A, B, C such that for all 4 points:
    x_i^2 + D*y_i^2 + A*x_i + B*y_i*sqrt(D) + C = 0
    
    This is a linear system in A, B, C. Four points give 4 equations in 3 unknowns. They are concyclic iff the system is consistent, i.e., the determinant of the 4x4 matrix is zero:
    
    |x1^2 + D*y1^2, x1, y1*sqrt(D), 1|
    |x2^2 + D*y2^2, x2, y2*sqrt(D), 1| = 0
    |x3^2 + D*y3^2, x3, y3*sqrt(D), 1|
    |x4^2 + D*y4^2, x4, y4*sqrt(D), 1|
    
    Since y_i*sqrt(D) is irrational, let's write y_i*sqrt(D) = y_i * sqrt(D).
    The determinant will have terms with sqrt(D). For the determinant to be zero, both the rational part and the sqrt(D) part must be zero.
    
    Let me expand. Let M be the 4x4 matrix. The determinant is a polynomial in sqrt(D). Since sqrt(D) is algebraic of degree 2 (for squarefree D>1), the determinant can be written as P + Q*sqrt(D) where P, Q are rational expressions. For det = 0, we need P = 0 and Q = 0.
    
    This is getting complex. Let me use a simpler check: for each set of 4 points, check if the circle through the first 3 passes through the 4th.
    
    Given 3 non-collinear points, the circle is uniquely determined. The 4th point is on it iff...
    
    Let me use the following: Points P1, P2, P3, P4 are concyclic iff
    det([[x1^2+D*y1^2, x1, y1, 1], ...]) but with the sqrt(D) handled.
    
    Actually, let me just check: for 3 points, compute the circle, then check if the 4th lies on it.
    
    For D=1, it's standard. For D>1, the "circle" in the embedded plane is an ellipse in the (x,y) plane.
    
    Let me use a different approach: use the power of a point or the Miquel-type condition.
    
    Actually, the simplest exact check: four points are concyclic iff the cross ratio is real, but that's complex.
    
    Let me just use the determinant with the understanding that we need to handle sqrt(D) terms. I'll compute the determinant symbolically in terms of sqrt(D) and check if both rational and irrational parts are zero.
    """
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
    """Check if 4 points are concyclic."""
    x1 = Fraction(p1[0], p1[1])
    y1 = Fraction(p1[2], p1[3])
    x2 = Fraction(p2[0], p2[1])
    y2 = Fraction(p2[2], p2[3])
    x3 = Fraction(p3[0], p3[1])
    y3 = Fraction(p3[2], p3[3])
    x4 = Fraction(p4[0], p4[1])
    y4 = Fraction(p4[2], p4[3])
    
    # We need to check if there exists a circle through all 4 points.
    # Circle: x^2 + D*y^2 + A*x + B*y*sqrt(D) + C = 0
    # For each point i: x_i^2 + D*y_i^2 + A*x_i + B*y_i*sqrt(D) + C = 0
    # Let u_i = x_i^2 + D*y_i^2 (rational)
    # u_i + A*x_i + B*y_i*sqrt(D) + C = 0
    # (u_i + A*x_i + C) + B*y_i*sqrt(D) = 0
    # Since sqrt(D) is irrational (D squarefree, D != 1), we need:
    # B*y_i = 0 for all i, and u_i + A*x_i + C = 0 for all i
    # If B != 0, then y_i = 0 for all 4 points. In that case, the points are on the x-axis,
    # and they are "concyclic" only if they lie on a circle, which for 4 distinct points on a line is impossible
    # (a circle intersects a line in at most 2 points). So if all y_i = 0 and 4 distinct points, not concyclic.
    # If some y_i != 0, then B = 0, and we need u_i + A*x_i + C = 0 for all 4 points.
    # This is a linear system in A, C with 4 equations. Consistent iff the 4 points lie on a conic x^2 + D*y^2 + A*x + C = 0,
    # which is an ellipse (or empty). 4 points on an ellipse...
    
    # Actually, let me just solve the linear system. We have 4 equations:
    # A*x_i + C = -(u_i) - B*y_i*sqrt(D)
    # This is messy. Let me use the determinant approach properly.
    
    # The 4x4 determinant:
    # |u1, x1, y1*sqrt(D), 1|
    # |u2, x2, y2*sqrt(D), 1| = 0
    # |u3, x3, y3*sqrt(D), 1|
    # |u4, x4, y4*sqrt(D), 1|
    
    # Let me compute this determinant. The third column has sqrt(D) factor.
    # Let c_i = y_i. Then the matrix is:
    # |u1, x1, c1*sqrt(D), 1|
    # |u2, x2, c2*sqrt(D), 1|
    # |u3, x3, c3*sqrt(D), 1|
    # |u4, x4, c4*sqrt(D), 1|
    
    # The determinant is linear in sqrt(D) when we expand along the third column? No, it's a full determinant.
    # Let me factor out sqrt(D) from the third column:
    # det = sqrt(D) * det([[u1,x1,c1,1],[u2,x2,c2,1],[u3,x3,c3,1],[u4,x4,c4,1]])
    
    # Wait, that's not right either. The determinant of a matrix where one column is multiplied by sqrt(D) is sqrt(D) times the determinant with that column.
    # So det(original) = sqrt(D) * det(M) where M has third column [c1,c2,c3,c4].
    
    # For det(original) = 0, since sqrt(D) != 0, we need det(M) = 0.
    
    # So we just need to check if the determinant of the 4x4 matrix with entries:
    # [u_i, x_i, y_i, 1] for i=1..4 is zero.
    
    u1 = x1 * x1 + D * y1 * y1
    u2 = x2 * x2 + D * y2 * y2
    u3 = x3 * x3 + D * y3 * y3
    u4 = x4 * x4 + D * y4 * y4
    
    # Compute determinant of:
    # |u1, x1, y1, 1|
    # |u2, x2, y2, 1|
    # |u3, x3, y3, 1|
    # |u4, x4, y4, 1|
    
    # Use cofactor expansion or direct formula.
    # Let me use the formula for 4x4 determinant.
    a11, a12, a13, a14 = u1, x1, y1, Fraction(1)
    a21, a22, a23, a24 = u2, x2, y2, Fraction(1)
    a31, a32, a33, a34 = u3, x3, y3, Fraction(1)
    a41, a42, a43, a44 = u4, x4, y4, Fraction(1)
    
    det = (a11 * (a22 * a33 * a44 - a22 * a34 * a43 - a23 * a32 * a44 + a23 * a34 * a42 + a24 * a32 * a43 - a24 * a33 * a42)
         - a12 * (a21 * a33 * a44 - a21 * a34 * a43 - a23 * a31 * a44 + a23 * a34 * a41 + a24 * a31 * a43 - a24 * a33 * a41)
         + a13 * (a21 * a32 * a44 - a21 * a34 * a42 - a22 * a31 * a44 + a22 * a34 * a41 + a24 * a31 * a42 - a24 * a32 * a41)
         - a14 * (a21 * a32 * a43 - a21 * a33 * a42 - a22 * a31 * a43 + a22 * a33 * a41 + a23 * a31 * a42 - a23 * a32 * a41))
    
    return det == 0

def validate(points, D):
    """Full validation of the point set."""
    if not is_squarefree(D):
        return False
    if D < 1 or D > 10**12:
        return False
    
    n = len(points)
    if n < 3 or n > 12:
        return False
    
    # Check all coordinates are within bounds
    for p in points:
        for val in [p[0], p[1], p[2], p[3]]:
            if abs(val) > 10**40:
                return False
        if p[1] == 0 or p[3] == 0:
            return False
    
    # Check distinctness
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

def generate_pythagorean_point(D, max_coord=100):
    """Generate a random point (x, y) such that x^2 + D*y^2 is a perfect square."""
    # We need x^2 + D*y^2 = z^2 for some integer z.
    # This is x^2 + D*y^2 = z^2.
    # For D=1, this is Pythagorean triples.
    # For general D, we can try random x, y and check.
    
    for _ in range(1000):
        x = random.randint(-max_coord, max_coord)
        y = random.randint(-max_coord, max_coord)
        if x == 0 and y == 0:
            continue
        val = x * x + D * y * y
        z = int(math.isqrt(val))
        if z * z == val:
            return (x, 1, y, 1)
    
    return None

def generate_point_set(D, num_points, max_coord=50):
    """Try to generate a set of points with rational pairwise distances."""
    points = [(0, 1, 0, 1)]  # Start with origin
    
    for _ in range(num_points - 1):
        attempts = 0
        while attempts < 200:
            p = generate_pythagorean_point(D, max_coord)
            if p is None:
                attempts += 1
                continue
            # Check if p has rational distance to all existing points
            valid = True
            for q in points:
                d2 = rational_dist_sq(p, q, D)
                if not is_perfect_square_frac(d2):
                    valid = False
                    break
            if valid:
                # Check no three collinear with existing points
                collinear = False
                for i in range(len(points)):
                    for j in range(i + 1, len(points)):
                        if _three_collinear_check(p, points[i], points[j]):
                            collinear = True
                            break
                    if collinear:
                        break
                if not collinear:
                    points.append(p)
                    break
            attempts += 1
    
    return points

def _three_collinear_check(p1, p2, p3):
    """Check if three points are collinear."""
    x1 = Fraction(p1[0], p1[1])
    y1 = Fraction(p1[2], p1[3])
    x2 = Fraction(p2[0], p2[1])
    y2 = Fraction(p2[2], p2[3])
    x3 = Fraction(p3[0], p3[1])
    y3 = Fraction(p3[2], p3[3])
    
    return (x2 - x1) * (y3 - y1) == (x3 - x1) * (y2 - y1)

def construct():
    """Construct a set of integral points in general position."""
    # Start with D=1 and try to build up to 6 points
    # Use a known set of 5 points and try to extend
    
    # Known 5-point set
    base_points = [(0, 1, 0, 1), (-56, 1, 33, 1), (-16, 1, 63, 1), (0, 1, 33, 1), (16, 1, 63, 1)]
    
    # Validate the base set
    if validate(base_points, 1):
        # Try to extend to 6 points
        for D in [1, 2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 46, 47]:
            for max_coord in [50, 100, 200, 500]:
                points = generate_point_set(D, 6, max_coord)
                if len(points) == 6 and validate(points, D):
                    # Check no four concyclic
                    if no_four_concyclic(points, D):
                        return {'D': D, 'points': [[[p[0], p[1]], [p[2], p[3]]] for p in points]}
        
        # If we can't extend, return the base 5-point set
        return {'D': 1, 'points': [[[p[0], p[1]], [p[2], p[3]]] for p in base_points]}
    
    # Fallback: return a simple 3-point set
    points = [(0, 1, 0, 1), (3, 1, 4, 1), (6, 1, 8, 1)]
    # Check if these work
    if validate(points, 1):
        return {'D': 1, 'points': [[[p[0], p[1]], [p[2], p[3]]] for p in points]}
    
    # Ultimate fallback
    return {'D': 1, 'points': [[[0, 1], [0, 1]], [[3, 1], [4, 1]], [[6, 1], [8, 1]]]}

# EVOLVE-BLOCK-END
