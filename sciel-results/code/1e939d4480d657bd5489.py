import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    def is_square(n):
        if n < 0:
            return False
        s = math.isqrt(n)
        return s * s == n

    def check_rational_dist(dx_num, dx_den, dy_num, dy_den, D):
        """Check if (dx_num/dx_den)^2 + D*(dy_num/dy_den)^2 is a perfect square of a rational."""
        # dist^2 = (dx_num/dx_den)^2 + D*(dy_num/dy_den)^2
        # = (dx_num^2 * dy_den^2 + D * dy_num^2 * dx_den^2) / (dx_den^2 * dy_den^2)
        num = dx_num * dx_num * dy_den * dy_den + D * dy_num * dy_num * dx_den * dx_den
        den = dx_den * dx_den * dy_den * dy_den
        if den == 0:
            return False
        if num == 0:
            return True  # distance 0, but we'll check distinctness separately
        # Check if num/den is a perfect square of a rational
        # num/den = (a/b)^2 means num*b^2 = a^2*den... no
        # num/den = a^2/b^2 means num*b^2 = a^2*den
        # So num/den must be a square of a rational.
        # Equivalently, num * den must be a perfect square (for positive num, den)
        # Actually, num/den is a perfect square iff num * den is a perfect square (when num, den > 0)
        # More precisely: num/den = p^2/q^2 in lowest terms iff num*q^2 = p^2*den
        # This is equivalent to num*den being a perfect square (after reducing)
        # Let's just check if num*den is a perfect square
        prod = num * den
        if prod < 0:
            return False
        return is_square(prod)

    def check_general_position(points, D):
        """Check no three collinear and no four concyclic. Points are (x_num, x_den, y_num, y_den)."""
        n = len(points)
        
        # Helper: get Fraction coords
        def get_frac(i):
            p = points[i]
            return Fraction(p[0], p[1]), Fraction(p[2], p[3])
        
        # Check no three collinear
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    x1, y1 = get_frac(i)
                    x2, y2 = get_frac(j)
                    x3, y3 = get_frac(k)
                    # Cross product (x2-x1)(y3-y1) - (y2-y1)(x3-x1)
                    cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                    if cross == 0:
                        return False
        
        # Check no four concyclic
        # A circle in Q(sqrt(D)) has equation x^2 + D*y^2 + A*x + B*y + C = 0
        # where A, B, C are in Q(sqrt(D))
        # For 4 points, we need to check if they're concyclic.
        # Use the determinant condition:
        # | x^2+Dy^2, x, y, 1 | = 0 for all 4 points
        # | x1^2+Dy1^2, x1, y1, 1 |
        # ...
        
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        p1 = points[i]
                        p2 = points[j]
                        p3 = points[k]
                        p4 = points[l]
                        
                        x1 = Fraction(p1[0], p1[1])
                        y1 = Fraction(p1[2], p1[3])
                        x2 = Fraction(p2[0], p2[1])
                        y2 = Fraction(p2[2], p2[3])
                        x3 = Fraction(p3[0], p3[1])
                        y3 = Fraction(p3[2], p3[3])
                        x4 = Fraction(p4[0], p4[1])
                        y4 = Fraction(p4[2], p4[3])
                        
                        # Compute x^2 + D*y^2 for each
                        v1 = x1 * x1 + D * y1 * y1
                        v2 = x2 * x2 + D * y2 * y2
                        v3 = x3 * x3 + D * y3 * y3
                        v4 = x4 * x4 + D * y4 * y4
                        
                        # 4x4 determinant
                        # | v1, x1, y1, 1 |
                        # | v2, x2, y2, 1 |
                        # | v3, x3, y3, 1 |
                        # | v4, x4, y4, 1 |
                        
                        # Subtract row 4 from rows 1,2,3
                        # | v1-v4, x1-x4, y1-y4, 0 |
                        # | v2-v4, x2-x4, y2-y4, 0 |
                        # | v3-v4, x3-x4, y3-y4, 0 |
                        # | v4,    x4,    y4,    1 |
                        
                        # Determinant = v4*det3 - x4*det3' + y4*det3'' - det3'''
                        # Where det3 is the 3x3 determinant of the first 3 columns of the first 3 rows
                        # Actually, expanding along the last column:
                        # det = 1 * det([[v1-v4, x1-x4, y1-y4],
                        #                  [v2-v4, x2-x4, y2-y4],
                        #                  [v3-v4, x3-x4, y3-y4]])
                        
                        a11 = v1 - v4
                        a12 = x1 - x4
                        a13 = y1 - y4
                        a21 = v2 - v4
                        a22 = x2 - x4
                        a23 = y2 - y4
                        a31 = v3 - v4
                        a32 = x3 - x4
                        a33 = y3 - y4
                        
                        det3 = (a11 * (a22 * a33 - a23 * a32)
                               - a12 * (a21 * a33 - a23 * a31)
                               + a13 * (a21 * a32 - a22 * a31))
                        
                        if det3 == 0:
                            return False
        
        return True

    def is_valid_set(points, D):
        """Full validity check."""
        n = len(points)
        if n < 3:
            return False
        
        # Check distinctness
        for i in range(n):
            for j in range(i + 1, n):
                p1 = points[i]
                p2 = points[j]
                if Fraction(p1[0], p1[1]) == Fraction(p2[0], p2[1]) and \
                   Fraction(p1[2], p1[3]) == Fraction(p2[2], p2[3]):
                    return False
        
        # Check rational distances
        for i in range(n):
            for j in range(i + 1, n):
                p1 = points[i]
                p2 = points[j]
                x1 = Fraction(p1[0], p1[1])
                y1 = Fraction(p1[2], p1[3])
                x2 = Fraction(p2[0], p2[1])
                y2 = Fraction(p2[2], p2[3])
                dx = x2 - x1
                dy = y2 - y1
                if not check_rational_dist(dx.numerator, dx.denominator, dy.numerator, dy.denominator, D):
                    return False
        
        # Check general position
        if not check_general_position(points, D):
            return False
        
        return True

    def try_add_point(points, D, max_coord_bound):
        """Try to add a 6th point to a 5-point set by searching nearby integer points."""
        if len(points) != 5:
            return None
        
        # Get bounding box
        xs = [Fraction(p[0], p[1]) for p in points]
        ys = [Fraction(p[2], p[3]) for p in points]
        
        min_x = min(int(math.floor(x)) for x in xs)
        max_x = max(int(math.ceil(x)) for x in xs)
        min_y = min(int(math.floor(y)) for y in ys)
        max_y = max(int(math.ceil(y)) for y in ys)
        
        # Expand search range
        expand = max_coord_bound
        search_min_x = min_x - expand
        search_max_x = max_x + expand
        search_min_y = min_y - expand
        search_max_y = max_y + expand
        
        # Limit total search points
        max_points_to_try = 50000
        
        # Create a set for quick lookup
        existing = set()
        for p in points:
            existing.add((Fraction(p[0], p[1]), Fraction(p[2], p[3])))
        
        count = 0
        for x in range(search_min_x, search_max_x + 1):
            for y in range(search_min_y, search_max_y + 1):
                if count >= max_points_to_try:
                    return None
                count += 1
                
                new_point = (x, 1, y, 1)
                # Check distinctness
                if (Fraction(x, 1), Fraction(y, 1)) in existing:
                    continue
                
                # Check distances to all 5 points
                valid = True
                for i in range(5):
                    p = points[i]
                    px = Fraction(p[0], p[1])
                    py = Fraction(p[2], p[3])
                    dx = Fraction(x, 1) - px
                    dy = Fraction(y, 1) - py
                    if not check_rational_dist(dx.numerator, dx.denominator, dy.numerator, dy.denominator, D):
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Check general position for all 6 points
                new_points = points + [new_point]
                if check_general_position(new_points, D):
                    return new_point
        
        return None

    # Initial 5-point set (integral point set)
    # These are known integral point sets
    points_5 = [
        (0, 1, 0, 1),
        (-56, 1, 33, 1),
        (-16, 1, 63, 1),
        (0, 1, 33, 1),
        (16, 1, 63, 1)
    ]
    
    D = 1
    
    # Verify the initial set is valid
    assert is_valid_set(points_5, D), "Initial 5-point set is not valid!"
    
    best_points = points_5
    
    # Try to find a 6th point with D=1
    if time_left() > 10:
        p6 = try_add_point(points_5, D, 200)
        if p6 is not None:
            candidate = points_5 + [p6]
            if is_valid_set(candidate, D):
                best_points = candidate
    
    # If still 5 points, try with D=3 (quadratic field)
    if len(best_points) == 5 and time_left() > 10:
        # Try a different base set in Q(sqrt(3))
        # Points: (0,0), (1,0), (0,1/sqrt(3)) -> not integer
        # Let's use points with rational coordinates in Q(sqrt(3))
        # Represent as (x_num, x_den, y_num, y_den) where y is multiplied by sqrt(D)
        
        # Try small integer points with D=3
        # (0,0), (3,0), (0,1): dist^2 = 9, 3, 12 - not all squares
        # Let's search for a valid 5-point set with D=3
        
        D_test = 3
        found_5 = False
        base_points_D3 = []
        
        # Try to find 5 points in Q(sqrt(3))
        # Start with (0,0) and (1,0)
        # dist^2 = 1, OK
        base = [(0, 1, 0, 1), (1, 1, 0, 1)]
        
        # Add (0,1): dist^2 to (0,0) = 3, to (1,0) = 1+3=4=2^2, OK
        base.append((0, 1, 1, 1))
        
        # Add (1,1): dist^2 to (0,0) = 1+3=4, to (1,0) = 3, to (0,1) = 1+0=1
        # 3 is not a square. So (1,1) doesn't work.
        
        # Try (2,1): dist^2 to (0,0) = 4+3=7, not a square.
        
        # Try (3,0): dist^2 to (0,0)=9, to (1,0)=4, to (0,1)=9+3=12, not a square.
        
        # This is getting complicated. Let's stick with D=1 and try harder.
        pass
    
    # Try randomized search for a 6th point with D=1, larger range
    if len(best_points) == 5 and time_left() > 20:
        random.seed(42)
        base = points_5
        bx = [Fraction(p[0], p[1]) for p in base]
        by = [Fraction(p[2], p[3]) for p in base]
        cx = sum(bx) / len(bx)
        cy = sum(by) / len(by)
        
        max_tries = 100000
        for trial in range(max_tries):
            if time_left() < 5:
                break
            # Sample randomly in a larger range
            x = int(random.gauss(int(cx), 300))
            y = int(random.gauss(int(cy), 300))
            
            new_point = (x, 1, y, 1)
            
            # Quick check: is it distinct?
            distinct = True
            for p in base:
                if Fraction(p[0], p[1]) == Fraction(x, 1) and Fraction(p[2], p[3]) == Fraction(y, 1):
                    distinct = False
                    break
            if not distinct:
                continue
            
            # Check distances
            valid = True
            for p in base:
                px = Fraction(p[0], p[1])
                py = Fraction(p[2], p[3])
                dx = Fraction(x, 1) - px
                dy = Fraction(y, 1) - py
                if not check_rational_dist(dx.numerator, dx.denominator, dy.numerator, dy.denominator, D):
                    valid = False
                    break
            if not valid:
                continue
            
            # Check general position
            new_points = base + [new_point]
            if check_general_position(new_points, D):
                best_points = new_points
                break
    
    # Final validation
    final_D = D
    final_points = best_points
    
    # Convert to the required format
    result_points = []
    for p in final_points:
        x_frac = Fraction(p[0], p[1])
        y_frac = Fraction(p[2], p[3])
        result_points.append([[x_frac.numerator, x_frac.denominator], [y_frac.numerator, y_frac.denominator]])
    
    return {'D': final_D, 'points': result_points}
