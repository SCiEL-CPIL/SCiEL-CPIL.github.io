"""
Search for integral point sets in general position.
Points are (x, y*sqrt(D)) with rational x, y.
Distances must be rational.
No three collinear, no four concyclic.
Maximize number of points.
"""

from fractions import Fraction
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    # Helper functions for exact arithmetic with Fractions
    def is_square(frac):
        """Check if a non-negative Fraction is a perfect square of a rational."""
        if frac < 0:
            return False
        if frac == 0:
            return True
        n, d = frac.numerator, frac.denominator
        # n and d must be perfect squares
        sqrt_n = math.isqrt(n)
        sqrt_d = math.isqrt(d)
        return sqrt_n * sqrt_n == n and sqrt_d * sqrt_d == d

    def dist_sq(p1, p2, D):
        """Squared distance between two points (x1, y1) and (x2, y2) in Q x Q*sqrt(D)."""
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx * dx + D * dy * dy

    def check_rational_distances(pts, D):
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                d2 = dist_sq(pts[i], pts[j], D)
                if not is_square(d2):
                    return False
        return True

    def check_no_three_collinear(pts):
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    p1, p2, p3 = pts[i], pts[j], pts[k]
                    # Determinant of vectors p2-p1 and p3-p1
                    # (x2-x1)(y3-y1) - (y2-y1)(x3-x1) == 0
                    if (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p2[1] - p1[1]) * (p3[0] - p1[0]):
                        return False
        return True

    def check_no_four_concyclic(pts):
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        p1, p2, p3, p4 = pts[i], pts[j], pts[k], pts[l]
                        x1, y1 = p1
                        x2, y2 = p2
                        x3, y3 = p3
                        x4, y4 = p4
                        
                        # Circle equation: x^2 + D y^2 + Ax + By + C = 0
                        # This form is not standard for concyclicity in Euclidean plane unless D=1.
                        # For D=1, circle is x^2+y^2+Ax+By+C=0.
                        # For general D, the "circle" in the (x, y*sqrt(D)) plane is an ellipse in (x,y).
                        # However, the condition "no four concyclic" refers to the Euclidean plane embedding.
                        # The points are (x_i, y_i * sqrt(D)).
                        # Let X_i = x_i, Y_i = y_i * sqrt(D).
                        # We need to check if (X_i, Y_i) are concyclic.
                        # This is equivalent to checking if the points (x_i, y_i) lie on a conic of the form:
                        # A x^2 + B y^2 + C xy + D x + E y + F = 0?
                        # No, a circle in the (X, Y) plane is X^2 + Y^2 + A'X + B'Y + C' = 0.
                        # Substituting X=x, Y=y*sqrt(D):
                        # x^2 + D y^2 + A' x + B' y sqrt(D) + C' = 0.
                        # Since x, y, A', B', C' are rational (for rational points), and sqrt(D) is irrational (if D not square),
                        # the term B' y sqrt(D) must vanish or be absorbed?
                        # Actually, if D is not a perfect square, sqrt(D) is irrational.
                        # For the equation to hold for rational x,y, the coefficient of sqrt(D) must be 0?
                        # No, B' is a parameter of the circle.
                        # Let's use the determinant test for concyclicity of 4 points (X1,Y1)...(X4,Y4).
                        # Det [ X^2+Y^2, X, Y, 1 ] = 0.
                        # X^2+Y^2 = x^2 + D y^2.
                        # The entries are rational (x, y, x^2+D y^2).
                        # So we just compute the determinant of the matrix:
                        # M = [[x1^2 + D y1^2, x1, y1, 1],
                        #      [x2^2 + D y2^2, x2, y2, 1],
                        #      [x3^2 + D y3^2, x3, y3, 1],
                        #      [x4^2 + D y4^2, x4, y4, 1]]
                        # If det(M) == 0, they are concyclic.
                        
                        m11 = x1*x1 + D*y1*y1
                        m12 = x1
                        m13 = y1
                        m14 = 1
                        
                        m21 = x2*x2 + D*y2*y2
                        m22 = x2
                        m23 = y2
                        m24 = 1
                        
                        m31 = x3*x3 + D*y3*y3
                        m32 = x3
                        m33 = y3
                        m34 = 1
                        
                        m41 = x4*x4 + D*y4*y4
                        m42 = x4
                        m43 = y4
                        m44 = 1
                        
                        # Calculate determinant of 4x4 matrix
                        # Using Sarrus-like expansion or cofactor expansion
                        # det = m11*(m22*(m33*m44 - m34*m43) - m23*(m32*m44 - m34*m42) + m24*(m32*m43 - m33*m42))
                        #     - m12*(m21*(m33*m44 - m34*m43) - m23*(m31*m44 - m34*m41) + m24*(m31*m43 - m33*m41))
                        #     + m13*(m21*(m32*m44 - m34*m42) - m22*(m31*m44 - m34*m41) + m24*(m31*m42 - m32*m41))
                        #     - m14*(m21*(m32*m43 - m33*m42) - m22*(m31*m43 - m33*m41) + m23*(m31*m42 - m32*m41))
                        
                        det = (
                            m11 * (m22*(m33*m44 - m34*m43) - m23*(m32*m44 - m34*m42) + m24*(m32*m43 - m33*m42))
                            - m12 * (m21*(m33*m44 - m34*m43) - m23*(m31*m44 - m34*m41) + m24*(m31*m43 - m33*m41))
                            + m13 * (m21*(m32*m44 - m34*m42) - m22*(m31*m44 - m34*m41) + m24*(m31*m42 - m32*m41))
                            - m14 * (m21*(m32*m43 - m33*m42) - m22*(m31*m43 - m33*m41) + m23*(m31*m42 - m32*m41))
                        )
                        if det == 0:
                            return False
        return True

    def is_valid(pts, D):
        if len(pts) < 3:
            return False
        # Check distinct
        seen = set()
        for p in pts:
            key = (p[0], p[1])
            if key in seen:
                return False
            seen.add(key)
        
        return (check_rational_distances(pts, D) and
                check_no_three_collinear(pts) and
                check_no_four_concyclic(pts))

    # Known valid 5-point set with D=1
    # Points: (0,0), (-56,33), (-16,63), (0,33), (16,63)
    # Distances are integers.
    base_points = [
        (Fraction(0), Fraction(0)),
        (Fraction(-56), Fraction(33)),
        (Fraction(-16), Fraction(63)),
        (Fraction(0), Fraction(33)),
        (Fraction(16), Fraction(63))
    ]
    D = 1
    
    # Verify base set
    if not is_valid(base_points, D):
        # Fallback to a simpler set if base fails (should not happen)
        base_points = [
            (Fraction(0), Fraction(0)),
            (Fraction(3), Fraction(4)),
            (Fraction(0), Fraction(5))
        ]
        if not is_valid(base_points, D):
             return {'D': 1, 'points': [[[0,1],[0,1]], [[3,1],[4,1]], [[0,1],[5,1]]]}

    best_points = list(base_points)
    
    # Search for 6th point
    # We search for integer coordinates (x, y) for D=1
    # This restricts x, y to integers, which is a subset of rationals.
    # If we find an integer point, it is a valid rational point.
    
    # To speed up, we can iterate x, y in a range.
    # Range: The base points are in x:[-56, 16], y:[0, 63].
    # Let's search in a box around this.
    
    max_coord = 200
    found_6 = None
    
    # Optimization: Precompute squares? No, just check dist_sq is square.
    
    for x in range(-max_coord, max_coord + 1):
        if time.time() > deadline:
            break
        x_frac = Fraction(x)
        for y in range(-max_coord, max_coord + 1):
            if time.time() > deadline:
                break
            y_frac = Fraction(y)
            candidate = (x_frac, y_frac)
            
            # Check if candidate is in base points
            if candidate in base_points:
                continue
                
            # Check distances to all base points
            valid = True
            for bp in base_points:
                d2 = dist_sq(candidate, bp, D)
                if not is_square(d2):
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Check geometric constraints
            candidate_set = base_points + [candidate]
            if check_no_three_collinear(candidate_set) and check_no_four_concyclic(candidate_set):
                found_6 = candidate
                break
        if found_6:
            break
            
    if found_6:
        best_points = base_points + [found_6]
        
    # If we found 6 points, try to find a 7th
    if len(best_points) == 6 and time.time() < deadline - 30:
        found_7 = None
        # Search range can be slightly smaller or same
        max_coord_7 = 100
        for x in range(-max_coord_7, max_coord_7 + 1):
            if time.time() > deadline:
                break
            x_frac = Fraction(x)
            for y in range(-max_coord_7, max_coord_7 + 1):
                if time.time() > deadline:
                    break
                y_frac = Fraction(y)
                candidate = (x_frac, y_frac)
                
                if candidate in best_points:
                    continue
                    
                valid = True
                for bp in best_points:
                    d2 = dist_sq(candidate, bp, D)
                    if not is_square(d2):
                        valid = False
                        break
                
                if not valid:
                    continue
                    
                candidate_set = best_points + [candidate]
                if check_no_three_collinear(candidate_set) and check_no_four_concyclic(candidate_set):
                    found_7 = candidate
                    break
            if found_7:
                break
                
        if found_7:
            best_points = best_points + [found_7]

    # Format output
    # points: [[[x_num, x_den], [y_num, y_den]], ...]
    out_points = []
    for p in best_points:
        x, y = p
        out_points.append([
            [x.numerator, x.denominator],
            [y.numerator, y.denominator]
        ])
        
    return {'D': D, 'points': out_points}

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res['points'])} points with D={res['D']}")
