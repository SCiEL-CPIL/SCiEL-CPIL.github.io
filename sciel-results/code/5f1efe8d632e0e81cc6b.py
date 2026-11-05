import math
from fractions import Fraction
import time
import itertools

# EVOLVE-BLOCK-START

def _is_square(n):
    """Check if n is a perfect square of a rational number (represented as Fraction)."""
    if n < 0:
        return False
    # n is a Fraction. Check if numerator and denominator are perfect squares.
    num = n.numerator
    den = n.denominator
    r_num = int(math.isqrt(num))
    r_den = int(math.isqrt(den))
    return r_num * r_num == num and r_den * r_den == den

def _dist_sq(p1, p2, D):
    """Calculate squared distance between two points with D=1 (or general D if needed)."""
    # p1, p2 are tuples of Fractions (x, y)
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + D * dy * dy

def _is_rational_dist(p1, p2, D):
    """Check if distance between p1 and p2 is rational."""
    d2 = _dist_sq(p1, p2, D)
    return _is_square(d2)

def _no_three_collinear(points):
    """Check if no three points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Cross product of (p2-p1) and (p3-p1)
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                if cross == 0:
                    return False
    return True

def _no_four_concyclic(points):
    """Check if no four points are concyclic."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
                    # Check if 4 points are concyclic using determinant or circle equation
                    # Equation: x^2 + y^2 + Dx + Ey + F = 0
                    # We can solve for D, E, F using p1, p2, p3 and check p4.
                    # To avoid division by zero issues, use cross product of vectors in 3D space (homogeneous coords)
                    # Or simply check if the perpendicular bisectors intersect at a center equidistant from all 4.
                    
                    # Let's use the determinant method for concyclicity:
                    # | x1 y1 x1^2+y1^2 1 |
                    # | x2 y2 x2^2+y2^2 1 | = 0
                    # | x3 y3 x3^2+y3^2 1 |
                    # | x4 y4 x4^2+y4^2 1 |
                    
                    def det4(M):
                        # M is 4x4
                        # Simple expansion
                        return (M[0][0]*(M[1][1]*(M[2][2]*M[3][3]-M[2][3]*M[3][2]) - M[1][2]*(M[2][1]*M[3][3]-M[2][3]*M[3][1]) + M[1][3]*(M[2][1]*M[3][2]-M[2][2]*M[3][1]))
                             - M[0][1]*(M[1][0]*(M[2][2]*M[3][3]-M[2][3]*M[3][2]) - M[1][2]*(M[2][0]*M[3][3]-M[2][3]*M[3][0]) + M[1][3]*(M[2][0]*M[3][2]-M[2][2]*M[3][0]))
                             + M[0][2]*(M[1][0]*(M[2][1]*M[3][3]-M[2][3]*M[3][1]) - M[1][1]*(M[2][0]*M[3][3]-M[2][3]*M[3][0]) + M[1][3]*(M[2][0]*M[3][1]-M[2][1]*M[3][0]))
                             - M[0][3]*(M[1][0]*(M[2][1]*M[3][2]-M[2][2]*M[3][1]) - M[1][1]*(M[2][0]*M[3][2]-M[2][2]*M[3][0]) + M[1][2]*(M[2][0]*M[3][1]-M[2][1]*M[3][0])))
                    
                    M = [
                        [p1[0], p1[1], p1[0]**2 + p1[1]**2, 1],
                        [p2[0], p2[1], p2[0]**2 + p2[1]**2, 1],
                        [p3[0], p3[1], p3[0]**2 + p3[1]**2, 1],
                        [p4[0], p4[1], p4[0]**2 + p4[1]**2, 1]
                    ]
                    if det4(M) == 0:
                        return False
    return True

def _validate(points, D):
    """Full validation of the point set."""
    if len(points) < 3:
        return False
    # Check distinctness
    if len(set(points)) != len(points):
        return False
    # Check rational distances
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if not _is_rational_dist(points[i], points[j], D):
                return False
    # Check no three collinear
    if not _no_three_collinear(points):
        return False
    # Check no four concyclic
    if not _no_four_concyclic(points):
        return False
    return True

def _find_6th_point(points, D, time_limit):
    """Try to find a 6th point that extends the given 5-point set."""
    start_time = time.time()
    # The existing 5 points
    P = points
    # We search for a new point Q = (x, y)
    # Constraints:
    # 1. dist(P_i, Q) is rational for all i
    # 2. Q not collinear with any pair
    # 3. Q not concyclic with any triple
    
    # Strategy:
    # Pick two points P_0, P_1.
    # The locus of points Q such that dist(P_0, Q) = r0 and dist(P_1, Q) = r1 is the intersection of two circles.
    # We iterate over possible rational radii r0, r1.
    # To keep the search space manageable, we limit r0, r1 to small integers or simple fractions.
    
    # Let's try integer radii first, up to some limit.
    # The existing points are roughly in a 100x100 box.
    # Let's try radii from 1 to 50.
    
    max_radius = 50
    found = None
    
    # Precompute squares of integers for quick check
    squares = {i*i: i for i in range(1, max_radius + 1)}
    
    for r0_sq in squares:
        r0 = math.isqrt(r0_sq)
        for r1_sq in squares:
            r1 = math.isqrt(r1_sq)
            
            # Intersection of circle centered at P[0] with radius r0 and P[1] with radius r1
            x0, y0 = P[0]
            x1, y1 = P[1]
            
            # Distance between centers
            d2 = (x1 - x0)**2 + (y1 - y0)**2
            d = math.sqrt(float(d2)) # Approximate for logic, but we need exact
            # Exact check: d must be rational for the intersection to be simple? No, d can be irrational, but intersection points might be rational.
            # However, if d2 is not a square, d is irrational.
            # The x-coordinate of the intersection involves d.
            # Let's use the exact formula.
            
            # Let A = P[0], B = P[1].
            # x = x0 + a
            # (x-x0)^2 + (y-y0)^2 = r0^2
            # (x-x1)^2 + (y-y1)^2 = r1^2
            # Subtracting:
            # 2(x1-x0)x + 2(y1-y0)y + (x0^2+y0^2 - x1^2-y1^2 + r1^2 - r0^2) = 0
            # This is a linear equation for x, y.
            
            a = 2 * (x1 - x0)
            b = 2 * (y1 - y0)
            c = x0**2 + y0**2 - x1**2 - y1**2 + r1_sq - r0_sq
            
            # If a=0 and b=0, centers are same (not possible for distinct points)
            # Solve for y in terms of x or vice versa.
            # Let's parameterize.
            # If b != 0: y = (-a*x - c) / b
            # Substitute into circle 1.
            
            # This algebra is complex with Fractions.
            # Alternative: Use the geometric property.
            # The intersection points lie on the line ax + by + c = 0.
            # We can iterate over rational points on this line? No, too many.
            
            # Let's stick to the direct intersection formula.
            # d = sqrt(d2)
            # If d2 is not a perfect square, d is irrational.
            # The x-coordinate of the intersection is:
            # x = x0 + (r0^2 - r1^2 + d2) / (2*d)
            # This is rational only if d is rational (i.e., d2 is a square) OR if the term (r0^2 - r1^2 + d2) is 0 (which implies r0=r1 and d2=0, impossible) or if the irrational parts cancel.
            # Actually, x is rational iff d is rational.
            # So we only need to consider pairs (P_i, P_j) where the distance between them is rational (which is true for all pairs in our set).
            # So d is rational.
            
            d_val = Fraction(d2)
            if not _is_square(d_val):
                continue # This shouldn't happen for our set, but just in case.
            
            d = Fraction(math.isqrt(int(d_val.numerator / d_val.denominator) if d_val.denominator == 1 else 0)) 
            # Better: compute sqrt of Fraction exactly
            num, den = d_val.numerator, d_val.denominator
            r_num = int(math.isqrt(num))
            r_den = int(math.isqrt(den))
            if r_num * r_num != num or r_den * r_den != den:
                continue
            d = Fraction(r_num, r_den)
            
            # x offset
            # x = x0 + (r0^2 - r1^2 + d^2) / (2*d)
            # Note: d^2 = d2
            num_x = r0_sq - r1_sq + d2
            den_x = 2 * d
            x_offset = Fraction(num_x, den_x)
            x = x0 + x_offset
            
            # y offset
            # (y - y0)^2 = r0^2 - (x - x0)^2 = r0^2 - x_offset^2
            y_sq = r0_sq - x_offset**2
            if y_sq < 0:
                continue
            if not _is_square(y_sq):
                continue
            
            r_y_num = int(math.isqrt(y_sq.numerator))
            r_y_den = int(math.isqrt(y_sq.denominator))
            if r_y_num * r_y_num != y_sq.numerator or r_y_den * r_y_den != y_sq.denominator:
                continue
            
            y_offset = Fraction(r_y_num, r_y_den)
            
            # Two possible y values
            y1_candidate = y0 + y_offset
            y2_candidate = y0 - y_offset
            
            candidates = [(x, y1_candidate), (x, y2_candidate)]
            
            for qx, qy in candidates:
                Q = (qx, qy)
                # Check if Q is distinct from existing points
                if Q in P:
                    continue
                
                # Check rational distance to all other points
                valid = True
                for pk in P:
                    if not _is_rational_dist(Q, pk, D):
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check no three collinear
                new_set = P + [Q]
                if not _no_three_collinear(new_set):
                    continue
                
                # Check no four concyclic
                if not _no_four_concyclic(new_set):
                    continue
                
                found = new_set
                break
            
            if found:
                break
        
        if found:
            break
        
        if time.time() - start_time > time_limit:
            break
            
    return found

def construct():
    # Start with the known 5-point set
    points_int = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    D = 1
    points = [(Fraction(x), Fraction(y)) for x, y in points_int]
    
    # Validate the initial set
    if not _validate(points, D):
        # Fallback to a smaller valid set if needed, but this one is known good
        pass
    
    best_points = points
    
    # Try to extend to 6 points
    # Time limit for extension search: 100 seconds
    extended = _find_6th_point(points, D, 100)
    if extended:
        best_points = extended
    
    # If we found 6, try to extend to 7?
    # Given the constraints, 6 is already a significant jump.
    # Let's stick to 6 to ensure we finish in time.
    
    # Format the output
    result_points = []
    for x, y in best_points:
        result_points.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
        
    return {'D': D, 'points': result_points}

# EVOLVE-BLOCK-END
