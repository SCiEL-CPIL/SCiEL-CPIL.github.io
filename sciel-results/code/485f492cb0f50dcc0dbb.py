import time
import math
from itertools import combinations
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 140.0

    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]

    def is_int_dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        d2 = dx*dx + dy*dy
        r = int(math.isqrt(d2))
        return r*r == d2

    def is_collinear(p1, p2, p3):
        return (p2[0]-p1[0])*(p3[1]-p1[1]) == (p2[1]-p1[1])*(p3[0]-p1[0])

    def is_concyclic(p1, p2, p3, p4):
        # Circle equation: x^2 + y^2 + Ax + By + C = 0
        # For point (x,y): x^2+y^2 + Ax + By + C = 0
        # Solve for A, B, C using 3 points, check 4th
        
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        # Linear system for A, B, C:
        # 2*x1*A + 2*y1*B + C = x1^2 + y1^2
        # 2*x2*A + 2*y2*B + C = x2^2 + y2^2
        # 2*x3*A + 2*y3*B + C = x3^2 + y3^2
        
        # Use Cramer's rule
        # Matrix:
        # [2x1, 2y1, 1]
        # [2x2, 2y2, 1]
        # [2x3, 2y3, 1]
        # RHS: [x1^2+y1^2, x2^2+y2^2, x3^2+y3^2]
        
        a11, a12, a13 = 2*x1, 2*y1, 1
        a21, a22, a23 = 2*x2, 2*y2, 1
        a31, a32, a33 = 2*x3, 2*y3, 1
        b1 = x1*x1 + y1*y1
        b2 = x2*x2 + y2*y2
        b3 = x3*x3 + y3*y3
        
        # Determinant
        det = a11*(a22*a33 - a23*a32) - a12*(a21*a33 - a23*a31) + a13*(a21*a32 - a22*a31)
        if det == 0:
            # Points are collinear or degenerate
            return False
        
        # A = det_A / det
        det_A = b1*(a22*a33 - a23*a32) - a12*(b2*a33 - a23*b3) + a13*(b2*a32 - a22*b3)
        A_num = det_A
        A_den = det
        
        # B = det_B / det
        det_B = a11*(b2*a33 - a23*b3) - b1*(a21*a33 - a23*a31) + a13*(a21*b3 - b2*a31)
        B_num = det_B
        B_den = det
        
        # C = det_C / det
        det_C = a11*(a22*b3 - b2*a32) - a12*(a21*b3 - b2*a31) + b1*(a21*a32 - a22*a31)
        C_num = det_C
        C_den = det
        
        # Check if point 4 lies on circle:
        # x4^2 + y4^2 + A*x4 + B*y4 + C = 0
        # Multiply by B_den*C_den to avoid fractions:
        # (x4^2 + y4^2)*B_den*C_den + A_num*x4*C_den + B_num*y4*B_den + C_num*B_den = 0
        
        lhs = (x4*x4 + y4*y4)*A_den*C_den + A_num*x4*B_den*C_den + B_num*y4*A_den*B_den + C_num*A_den*B_den
        
        return lhs == 0

    def find_extension_via_grid(points, max_range, deadline):
        n = len(points)
        min_x = min(p[0] for p in points) - max_range
        max_x = max(p[0] for p in points) + max_range
        min_y = min(p[1] for p in points) - max_range
        max_y = max(p[1] for p in points) + max_range

        point_set = set(points)

        for x in range(min_x, max_x + 1):
            if time.time() > deadline:
                return None
            for y in range(min_y, max_y + 1):
                p_new = (x, y)
                if p_new in point_set:
                    continue

                valid_dist = True
                for p in points:
                    if not is_int_dist(p_new, p):
                        valid_dist = False
                        break
                if not valid_dist:
                    continue

                valid_col = True
                for i in range(n):
                    for j in range(i+1, n):
                        if is_collinear(points[i], points[j], p_new):
                            valid_col = False
                            break
                    if not valid_col:
                        break
                if not valid_col:
                    continue

                valid_con = True
                for i in range(n):
                    for j in range(i+1, n):
                        for k in range(j+1, n):
                            if is_concyclic(points[i], points[j], points[k], p_new):
                                valid_con = False
                                break
                        if not valid_con:
                            break
                    if not valid_con:
                        break
                if not valid_con:
                    continue

                return p_new
        return None

    def circle_through(p1, p2, p3):
        """Return (A, B, C) such that x^2+y^2+Ax+By+C=0 passes through p1,p2,p3.
        Returns None if collinear."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        
        a11, a12, a13 = 2*x1, 2*y1, 1
        a21, a22, a23 = 2*x2, 2*y2, 1
        a31, a32, a33 = 2*x3, 2*y3, 1
        b1 = x1*x1 + y1*y1
        b2 = x2*x2 + y2*y2
        b3 = x3*x3 + y3*y3
        
        det = a11*(a22*a33 - a23*a32) - a12*(a21*a33 - a23*a31) + a13*(a21*a32 - a22*a31)
        if det == 0:
            return None
        
        det_A = b1*(a22*a33 - a23*a32) - a12*(b2*a33 - a23*b3) + a13*(b2*a32 - a22*b3)
        det_B = a11*(b2*a33 - a23*b3) - b1*(a21*a33 - a23*a31) + a13*(a21*b3 - b2*a31)
        det_C = a11*(a22*b3 - b2*a32) - a12*(a21*b3 - b2*a31) + b1*(a21*a32 - a22*a31)
        
        A = Fraction(det_A, det)
        B = Fraction(det_B, det)
        C = Fraction(det_C, det)
        
        return (A, B, C)

    def rational_points_on_circle(A, B, C, known_point, max_param=200):
        """Generate rational points on circle x^2+y^2+Ax+By+C=0 using rational parameterization.
        known_point is a point on the circle. Returns list of (Fraction, Fraction) points."""
        x0, y0 = known_point
        
        # Circle center: (-A/2, -B/2), radius^2 = (A/2)^2 + (B/2)^2 - C
        # Parameterize using line through known_point with rational slope t:
        # y - y0 = t(x - x0)
        # Substitute into circle equation to get quadratic in x, one root is x0,
        # other root gives new point.
        
        points = []
        for t_num in range(-max_param, max_param + 1):
            for t_den in range(1, max_param + 1):
                if math.gcd(abs(t_num), t_den) != 1:
                    continue
                t = Fraction(t_num, t_den)
                
                # Line: y = t(x - x0) + y0
                # x^2 + (t(x-x0)+y0)^2 + Ax + B(t(x-x0)+y0) + C = 0
                # Expand:
                # x^2 + t^2(x-x0)^2 + 2t(x-x0)(y0) + y0^2 + Ax + Bt(x-x0) + By0 + C = 0
                # This is a quadratic in x. One root is x=x0.
                # Let's compute the other root.
                
                # Coefficient of x^2: 1 + t^2
                # Coefficient of x: -2*t^2*x0 + 2*t*y0 + A + B*t
                # Constant: t^2*x0^2 - 2*t*x0*y0 + y0^2 + B*t*(-x0) + B*y0 + C
                #          = t^2*x0^2 - 2*t*x0*y0 + y0^2 - B*t*x0 + B*y0 + C
                
                a_coef = 1 + t*t
                b_coef = -2*t*t*x0 + 2*t*y0 + A + B*t
                c_coef = t*t*x0*x0 - 2*t*x0*y0 + y0*y0 - B*t*x0 + B*y0 + C
                
                # Roots: x = (-b ± sqrt(b^2 - 4ac)) / (2a)
                # Since x0 is a root, the other root is: x1 = c_coef / (a_coef * x0) ... 
                # Actually, product of roots = c/a, so x0 * x1 = c/a, x1 = c/(a*x0)
                # But x0 might be 0. Use sum of roots: x0 + x1 = -b/a, x1 = -b/a - x0
                
                if a_coef == 0:
                    continue
                    
                x1 = -b_coef / a_coef - x0
                y1 = t * (x1 - x0) + y0
                
                # Check if this is a valid point (not the known point itself)
                if x1 != x0 or y1 != y0:
                    points.append((x1, y1))
        
        return points

    def check_point_validity(p_new, points, deadline):
        """Check if p_new (tuple of Fractions or ints) is valid to add to points."""
        # Convert to common type for comparison
        n = len(points)
        
        # Check distance to all existing points
        for p in points:
            dx = p_new[0] - p[0]
            dy = p_new[1] - p[1]
            d2 = dx*dx + dy*dy
            # Check if d2 is a perfect square of a rational
            # d2 should be a rational number that is a square
            if d2 < 0:
                return False
            # d2 is a Fraction or int. Check if it's a perfect square.
            if isinstance(d2, Fraction):
                num = d2.numerator
                den = d2.denominator
                r_num = int(math.isqrt(num))
                r_den = int(math.isqrt(den))
                if r_num * r_num != num or r_den * r_den != den:
                    return False
            else:
                r = int(math.isqrt(d2))
                if r * r != d2:
                    return False
        
        # Check collinearity
        for i in range(n):
            for j in range(i+1, n):
                if is_collinear(points[i], points[j], p_new):
                    return False
        
        # Check concyclicity
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if is_concyclic(points[i], points[j], points[k], p_new):
                        return False
        
        return True

    def try_add_point(points, deadline):
        """Try to find a new point to add to the set. Returns new point or None."""
        n = len(points)
        
        # Strategy 1: Try circles through triples of existing points
        # For each triple, get the circle, then find rational points on it
        # that have integer/rational distances to all existing points.
        
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if time.time() > deadline:
                        return None
                    circle = circle_through(points[i], points[j], points[k])
                    if circle is None:
                        continue
                    A, B, C = circle
                    
                    # Try parameterization with known point = points[i]
                    candidates = rational_points_on_circle(A, B, C, points[i], max_param=50)
                    
                    for cand in candidates:
                        # Convert to check
                        # cand is (Fraction, Fraction)
                        # Check if all coordinates are "nice" (small numerator/denominator)
                        if abs(cand[0].numerator) > 10**6 or cand[0].denominator > 10**6:
                            continue
                        if abs(cand[1].numerator) > 10**6 or cand[1].denominator > 10**6:
                            continue
                        
                        # Check validity
                        if check_point_validity(cand, points, deadline):
                            return cand
        
        # Strategy 2: Grid search for integer points
        p6 = find_extension_via_grid(points, 300, deadline)
        if p6:
            return p6
        
        return None

    best_points = seed_points

    # Try to find 6th point
    p6 = try_add_point(best_points, deadline)
    if p6:
        best_points = best_points + [p6]
        # Try to find 7th point
        p7 = try_add_point(best_points, deadline)
        if p7:
            best_points = best_points + [p7]
            # Try to find 8th point
            p8 = try_add_point(best_points, deadline)
            if p8:
                best_points = best_points + [p8]

    # Convert all points to integer form if possible
    final_points = []
    for p in best_points:
        x, y = p
        if isinstance(x, Fraction):
            x = x.numerator, x.denominator
        else:
            x = x, 1
        if isinstance(y, Fraction):
            y = y.numerator, y.denominator
        else:
            y = y, 1
        final_points.append([x, y])

    return {'D': 1, 'points': final_points}
