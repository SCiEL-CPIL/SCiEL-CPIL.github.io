import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Initial 5-point set
    points_5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    def is_square(n):
        if n < 0:
            return False
        s = int(math.isqrt(n))
        return s * s == n

    def check_rational_dist_sq(dx, dy, D):
        # dist^2 = dx^2 + D*dy^2
        # We need dist to be rational.
        # If dx, dy are integers, dist^2 is integer.
        # So dist is rational iff dist^2 is a perfect square.
        val = dx * dx + D * dy * dy
        return is_square(val)

    def check_general_position(points):
        # No three collinear
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    # Cross product (x2-x1)(y3-y1) - (y2-y1)(x3-x1)
                    if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0:
                        return False
        # No four concyclic
        # Check all 4-tuples
        # Equation of circle: x^2 + y^2 + Dx + Ey + F = 0
        # For 4 points, the determinant of the matrix must be non-zero
        # Or use the condition that the cross ratio is not real? 
        # Simpler: Check if 4 points lie on a circle.
        # A circle is determined by 3 points. Check if the 4th lies on it.
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        x1, y1 = points[i]
                        x2, y2 = points[j]
                        x3, y3 = points[k]
                        x4, y4 = points[l]
                        
                        # Calculate circle through P1, P2, P3
                        # 2(a x1 + b y1) = x1^2 + y1^2 - (x3^2 + y3^2) ? No.
                        # Let circle be x^2 + y^2 + Ax + By + C = 0
                        # x1^2 + y1^2 + A x1 + B y1 + C = 0
                        # x2^2 + y2^2 + A x2 + B y2 + C = 0
                        # x3^2 + y3^2 + A x3 + B y3 + C = 0
                        
                        # Subtract P3 from P1 and P2:
                        # A(x1-x3) + B(y1-y3) = x3^2 + y3^2 - x1^2 - y1^2
                        # A(x2-x3) + B(y2-y3) = x3^2 + y3^2 - x2^2 - y2^2
                        
                        # Let R1 = x3^2 + y3^2 - x1^2 - y1^2
                        # Let R2 = x3^2 + y3^2 - x2^2 - y2^2
                        # Let dx1 = x1-x3, dy1 = y1-y3
                        # Let dx2 = x2-x3, dy2 = y2-y3
                        
                        # Determinant of coeff matrix:
                        det = dx1 * dy2 - dx2 * dy1
                        if det == 0:
                            continue # Collinear, already handled, but skip circle check for degenerate case
                        
                        # A = (R1*dy2 - R2*dy1) / det
                        # B = (dx1*R2 - dx2*R1) / det
                        
                        # Check if P4 satisfies:
                        # x4^2 + y4^2 + A x4 + B y4 + C = 0
                        # C = -x3^2 - y3^2 - A x3 - B y3
                        
                        # We can avoid division by checking:
                        # (x4^2 + y4^2 + A_num x4 / det + B_num y4 / det + C_num / det) == 0
                        # Multiply by det:
                        # det*(x4^2 + y4^2) + A_num*x4 + B_num*y4 + C_num == 0
                        
                        A_num = R1 * dy2 - R2 * dy1
                        B_num = dx1 * R2 - dx2 * R1
                        # C_num = -det*(x3^2 + y3^2) - A_num*x3 - B_num*y3
                        # Actually, C is not needed if we use the equation directly?
                        # x4^2 + y4^2 + (A_num/det)*x4 + (B_num/det)*y4 + C = 0
                        # det*(x4^2 + y4^2) + A_num*x4 + B_num*y4 + det*C = 0
                        # We need det*C.
                        # From P3: C = -x3^2 - y3^2 - A x3 - B y3
                        # det*C = -det*(x3^2 + y3^2) - A_num*x3 - B_num*y3
                        
                        val = det * (x4 * x4 + y4 * y4) + A_num * x4 + B_num * y4 - det * (x3 * x3 + y3 * y3) - A_num * x3 - B_num * y3
                        if val == 0:
                            return False
        return True

    def try_find_6th_point(base_points, D, max_coord):
        # Base points are integers
        # Search for integer point (x,y) such that dist to all base points is integer
        # and general position holds for the set of 6 points
        
        # Optimization: Precompute base point coords
        bx = [p[0] for p in base_points]
        by = [p[1] for p in base_points]
        
        # Center of search
        cx = sum(bx) // len(bx)
        cy = sum(by) // len(by)
        
        # We can restrict search range to be relative to the bounding box of base points
        min_x = min(bx)
        max_x = max(bx)
        min_y = min(by)
        max_y = max(by)
        
        # Expand range
        range_x = max(max_x - min_x, 1) * 2
        range_y = max(max_y - min_y, 1) * 2
        
        # Search in a box around the centroid, size determined by max_coord or range
        # Let's try a box of size 200x200 first, then expand if time permits
        # But to be safe, we iterate x from cx - R to cx + R
        
        R = max_coord
        
        for x in range(cx - R, cx + R + 1):
            for y in range(cy - R, cy + R + 1):
                # Check distances
                valid = True
                for i in range(len(base_points)):
                    dx = x - bx[i]
                    dy = y - by[i]
                    if not check_rational_dist_sq(dx, dy, D):
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check if point is distinct from base points
                # (Implicitly handled if distance > 0, but dist 0 is square)
                # We need distinct points.
                # If dist is 0, it's the same point. We skip dist 0?
                # The problem says "distinct points".
                # So we must ensure (x,y) is not in base_points.
                if (x, y) in base_points:
                    continue
                
                # Check general position
                new_points = base_points + [(x, y)]
                if check_general_position(new_points):
                    return (x, y)
        return None

    # Try D=1
    # Start with small range, expand if time allows
    R = 100
    while R <= 500:
        if time.time() > deadline:
            break
        p6 = try_find_6th_point(points_5, 1, R)
        if p6:
            points = points_5 + [p6]
            return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points]}
        R += 100

    # Try D=2 with smaller range
    if time.time() < deadline:
        R = 50
        while R <= 200:
            if time.time() > deadline:
                break
            p6 = try_find_6th_point(points_5, 2, R)
            if p6:
                points = points_5 + [p6]
                return {'D': 2, 'points': [[[x, 1], [y, 1]] for x, y in points]}
            R += 50

    # Fallback to 5 points
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points_5]}
