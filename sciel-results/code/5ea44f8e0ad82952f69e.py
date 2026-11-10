import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    def is_rational_square(val):
        """Check if val (Fraction) is a square of a rational number."""
        if val < 0:
            return False
        if val == 0:
            return True
        num = val.numerator
        den = val.denominator
        s_num = int(math.isqrt(num))
        if s_num * s_num != num:
            return False
        s_den = int(math.isqrt(den))
        if s_den * s_den != den:
            return False
        return True

    def sqrt_frac(f):
        """Return the rational square root of f, or None if f is not a perfect square of a rational."""
        if f < 0:
            return None
        if f == 0:
            return Fraction(0)
        num = f.numerator
        den = f.denominator
        s_num = int(math.isqrt(num))
        if s_num * s_num != num:
            return None
        s_den = int(math.isqrt(den))
        if s_den * s_den != den:
            return None
        return Fraction(s_num, s_den)

    def check_collinear(pts):
        """Check if any three points in pts are collinear. Returns True if valid (no 3 collinear)."""
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]
                    if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0:
                        return False
        return True

    def check_concyclic(D, pts):
        """Check if any four points in pts are concyclic. Returns True if valid (no 4 concyclic).
        
        Four points (x_i, y_i) with Y_i = y_i * sqrt(D) are concyclic iff they lie on a circle
        x^2 + Y^2 + Ax + BY + C = 0.
        
        Since Y_i = y_i * sqrt(D), the equation becomes:
        x_i^2 + D*y_i^2 + A*x_i + B*y_i*sqrt(D) + C = 0
        
        For this to hold for rational x_i, y_i, we need to separate rational and irrational parts.
        But actually, A, B, C can involve sqrt(D) if needed... No, A, B, C are real numbers.
        
        Better approach: Use the determinant condition. Four points (x1,Y1)..(x4,Y4) are concyclic iff
        the determinant of the matrix:
        | x1^2 + Y1^2, x1, Y1, 1 |
        | x2^2 + Y2^2, x2, Y2, 1 |
        | x3^2 + Y3^2, x3, Y3, 1 |
        | x4^2 + Y4^2, x4, Y4, 1 |
        is zero.
        
        With Y_i = y_i * sqrt(D), we have x_i^2 + Y_i^2 = x_i^2 + D*y_i^2 (rational).
        The matrix entries are:
        Row i: [x_i^2 + D*y_i^2, x_i, y_i*sqrt(D), 1]
        
        The determinant will be of the form a + b*sqrt(D) where a, b are rational.
        For the determinant to be zero, we need a = 0 AND b = 0.
        
        We can compute the determinant by treating sqrt(D) as a symbol.
        Let's expand the determinant. The entries involving sqrt(D) are only in column 3.
        
        Actually, let's use a different approach. Four points are concyclic iff the circle through
        three of them passes through the fourth. We can compute the circle through P1, P2, P3 and
        check if P4 lies on it.
        
        Circle through P1, P2, P3: x^2 + Y^2 + Ax + BY + C = 0
        This gives us a system of 3 equations in A, B, C.
        
        For P1: x1^2 + D*y1^2 + A*x1 + B*y1*sqrt(D) + C = 0
        For P2: x2^2 + D*y2^2 + A*x2 + B*y2*sqrt(D) + C = 0
        For P3: x3^2 + D*y3^2 + A*x3 + B*y3*sqrt(D) + C = 0
        
        Let's write this as:
        A*x1 + B*y1*sqrt(D) + C = -(x1^2 + D*y1^2)
        A*x2 + B*y2*sqrt(D) + C = -(x2^2 + D*y2^2)
        A*x3 + B*y3*sqrt(D) + C = -(x3^2 + D*y3^2)
        
        This is a linear system in A, B, C. We can solve it using Cramer's rule or Gaussian elimination.
        The matrix is:
        [x1, y1*sqrt(D), 1]
        [x2, y2*sqrt(D), 1]
        [x3, y3*sqrt(D), 1]
        
        The determinant of this 3x3 matrix is:
        x1*(y2*sqrt(D)*1 - 1*y3*sqrt(D)) - y1*sqrt(D)*(x2*1 - 1*x3) + 1*(x2*y3*sqrt(D) - x3*y2*sqrt(D))
        = x1*sqrt(D)*(y2-y3) - y1*sqrt(D)*(x2-x3) + sqrt(D)*(x2*y3 - x3*y2)
        = sqrt(D) * [x1*(y2-y3) - y1*(x2-x3) + x2*y3 - x3*y2]
        = sqrt(D) * [x1*y2 - x1*y3 - y1*x2 + y1*x3 + x2*y3 - x3*y2]
        
        Let's call the bracket term T. So det = sqrt(D) * T.
        If T = 0, the three points are collinear (in the (x,Y) plane), so no unique circle.
        Otherwise, we can solve for A, B, C.
        
        Then check if P4 satisfies the circle equation.
        """
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        x1, y1 = pts[i]
                        x2, y2 = pts[j]
                        x3, y3 = pts[k]
                        x4, y4 = pts[l]
                        
                        # Check if P1, P2, P3 are collinear (in (x, Y) plane)
                        # Cross product: (x2-x1)*(y3*sqrt(D)-y1*sqrt(D)) - (y2*sqrt(D)-y1*sqrt(D))*(x3-x1)
                        # = sqrt(D) * [(x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)]
                        T = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                        if T == 0:
                            # Three points collinear, so the fourth can't make a concyclic set
                            # (unless all four are collinear, but we already checked no 3 collinear)
                            continue
                        
                        # Solve for A, B, C
                        # Matrix:
                        # [x1, y1*sqrt(D), 1] [A]   [-(x1^2+D*y1^2)]
                        # [x2, y2*sqrt(D), 1] [B] = [-(x2^2+D*y2^2)]
                        # [x3, y3*sqrt(D), 1] [C]   [-(x3^2+D*y3^2)]
                        
                        # Let's use Cramer's rule.
                        # det_main = sqrt(D) * T (as computed above)
                        # For A: replace column 1 with RHS
                        # det_A = determinant of:
                        # [-(x1^2+D*y1^2), y1*sqrt(D), 1]
                        # [-(x2^2+D*y2^2), y2*sqrt(D), 1]
                        # [-(x3^2+D*y3^2), y3*sqrt(D), 1]
                        # = -(x1^2+D*y1^2)*(y2*sqrt(D)*1 - 1*y3*sqrt(D)) + y1*sqrt(D)*(-(x2^2+D*y2^2)*1 - 1*(-(x3^2+D*y3^2))) - 1*(-(x2^2+D*y2^2)*y3*sqrt(D) - (-(x3^2+D*y3^2))*y2*sqrt(D))
                        # This is getting messy. Let's use a different approach.
                        
                        # Alternative: Use the fact that the circle equation can be written as:
                        # x^2 + D*y^2 + A*x + B*y*sqrt(D) + C = 0
                        # Let's define for each point (x,y):
                        # f(x,y) = x^2 + D*y^2
                        # The circle equation is: f(x,y) + A*x + B*y*sqrt(D) + C = 0
                        # 
                        # For four points to be concyclic, there must exist A, B, C such that:
                        # f(x_i, y_i) + A*x_i + B*y_i*sqrt(D) + C = 0 for i=1,2,3,4
                        # 
                        # This is equivalent to the 4x4 system:
                        # [x1, y1*sqrt(D), 1, f(x1,y1)]
                        # [x2, y2*sqrt(D), 1, f(x2,y2)]
                        # [x3, y3*sqrt(D), 1, f(x3,y3)]
                        # [x4, y4*sqrt(D), 1, f(x4,y4)]
                        # having a non-trivial solution for [A, B, C, -1] (or similar).
                        # 
                        # Actually, the condition is that the 4 points are linearly dependent in the
                        # vector space spanned by [x, y*sqrt(D), 1, f(x,y)].
                        # 
                        # Let's compute the determinant of the 4x4 matrix:
                        # M = [x_i, y_i*sqrt(D), 1, x_i^2 + D*y_i^2] for i=1,2,3,4
                        # 
                        # This determinant will be of the form a + b*sqrt(D).
                        # For the points to be concyclic, we need this determinant to be 0.
                        # So we need a = 0 and b = 0.
                        # 
                        # Let's compute this determinant by expanding along the column with sqrt(D).
                        # Column 2 is [y1*sqrt(D), y2*sqrt(D), y3*sqrt(D), y4*sqrt(D)] = sqrt(D) * [y1, y2, y3, y4]
                        # 
                        # det(M) = sqrt(D) * det(M') where M' is obtained by replacing column 2 with [y1, y2, y3, y4]
                        # and then multiplying by (-1)^(2+j) for each term... 
                        # 
                        # Actually, it's easier to just compute the determinant symbolically.
                        # Let's write the matrix as:
                        # [x1, y1*s, 1, x1^2+D*y1^2]
                        # [x2, y2*s, 1, x2^2+D*y2^2]
                        # [x3, y3*s, 1, x3^2+D*y3^2]
                        # [x4, y4*s, 1, x4^2+D*y4^2]
                        # where s = sqrt(D)
                        # 
                        # Expand along the last column (column 4):
                        # det = (x1^2+D*y1^2)*C14 - (x2^2+D*y2^2)*C24 + (x3^2+D*y3^2)*C34 - (x4^2+D*y4^2)*C44
                        # where Cij is the cofactor.
                        # 
                        # This is still complex. Let's use a numerical approach with high precision?
                        # No, we need exact arithmetic.
                        # 
                        # Alternative approach: Use the power of a point theorem or cross ratio.
                        # 
                        # Actually, the simplest exact approach: 
                        # Four points are concyclic iff the circle through any three passes through the fourth.
                        # We can compute the circle through P1, P2, P3 by solving the linear system,
                        # then check if P4 lies on it.
                        # 
                        # The system is:
                        # A*x1 + B*y1*s + C = -f1
                        # A*x2 + B*y2*s + C = -f2
                        # A*x3 + B*y3*s + C = -f3
                        # 
                        # where f_i = x_i^2 + D*y_i^2
                        # 
                        # We can solve this using Cramer's rule. The determinant of the coefficient matrix is:
                        # det_coeff = x1*(y2*s*1 - 1*y3*s) - y1*s*(x2*1 - 1*x3) + 1*(x2*y3*s - x3*y2*s)
                        # = x1*s*(y2-y3) - y1*s*(x2-x3) + s*(x2*y3 - x3*y2)
                        # = s * [x1*(y2-y3) - y1*(x2-x3) + x2*y3 - x3*y2]
                        # = s * T (as before)
                        # 
                        # For A: replace column 1 with RHS [-f1, -f2, -f3]
                        # det_A = (-f1)*(y2*s*1 - 1*y3*s) - y1*s*((-f2)*1 - 1*(-f3)) + 1*((-f2)*y3*s - (-f3)*y2*s)
                        # = (-f1)*s*(y2-y3) - y1*s*(f3-f2) + s*(f3*y2 - f2*y3)
                        # = s * [-f1*(y2-y3) - y1*(f3-f2) + f3*y2 - f2*y3]
                        # 
                        # So A = det_A / det_coeff = [-f1*(y2-y3) - y1*(f3-f2) + f3*y2 - f2*y3] / T
                        # 
                        # Similarly for B: replace column 2 with RHS
                        # det_B = x1*((-f2)*1 - 1*(-f3)) - (-f1)*(x2*1 - 1*x3) + 1*(x2*(-f3) - x3*(-f2))
                        # = x1*(f3-f2) + f1*(x2-x3) + x2*(-f3) + x3*f2
                        # = x1*f3 - x1*f2 + f1*x2 - f1*x3 - x2*f3 + x3*f2
                        # 
                        # So B = det_B / det_coeff = [x1*f3 - x1*f2 + f1*x2 - f1*x3 - x2*f3 + x3*f2] / (s*T)
                        # 
                        # Note that B involves 1/s = 1/sqrt(D). So B = B_rational / sqrt(D) where B_rational is rational.
                        # 
                        # For C: replace column 3 with RHS
                        # det_C = x1*((-f2)*y3*s - (-f3)*y2*s) - y1*s*(x2*(-f3) - x3*(-f2)) + (-f1)*(x2*y3*s - x3*y2*s)
                        # = x1*s*(f3*y2 - f2*y3) - y1*s*(-x2*f3 + x3*f2) - f1*s*(x2*y3 - x3*y2)
                        # = s * [x1*(f3*y2 - f2*y3) - y1*(-x2*f3 + x3*f2) - f1*(x2*y3 - x3*y2)]
                        # 
                        # So C = det_C / det_coeff = [x1*(f3*y2 - f2*y3) - y1*(-x2*f3 + x3*f2) - f1*(x2*y3 - x3*y2)] / T
                        # 
                        # Now, check if P4 lies on the circle:
                        # A*x4 + B*y4*s + C = -f4
                        # 
                        # Substitute A, B, C:
                        # (A_num/T)*x4 + (B_num/(s*T))*y4*s + (C_num/T) = -f4
                        # (A_num*x4 + B_num*y4 + C_num)/T = -f4
                        # A_num*x4 + B_num*y4 + C_num = -f4*T
                        # 
                        # Where:
                        # A_num = -f1*(y2-y3) - y1*(f3-f2) + f3*y2 - f2*y3
                        # B_num = x1*f3 - x1*f2 + f1*x2 - f1*x3 - x2*f3 + x3*f2
                        # C_num = x1*(f3*y2 - f2*y3) - y1*(-x2*f3 + x3*f2) - f1*(x2*y3 - x3*y2)
                        # T = x1*(y2-y3) - y1*(x2-x3) + x2*y3 - x3*y2
                        # 
                        # All of these are rational! So we can check:
                        # A_num*x4 + B_num*y4 + C_num + f4*T == 0
                        # 
                        f1 = x1*x1 + D*y1*y1
                        f2 = x2*x2 + D*y2*y2
                        f3 = x3*x3 + D*y3*y3
                        f4 = x4*x4 + D*y4*y4
                        
                        A_num = -f1*(y2-y3) - y1*(f3-f2) + f3*y2 - f2*y3
                        B_num = x1*f3 - x1*f2 + f1*x2 - f1*x3 - x2*f3 + x3*f2
                        C_num = x1*(f3*y2 - f2*y3) - y1*(-x2*f3 + x3*f2) - f1*(x2*y3 - x3*y2)
                        T = x1*(y2-y3) - y1*(x2-x3) + x2*y3 - x3*y2
                        
                        if A_num*x4 + B_num*y4 + C_num + f4*T != 0:
                            continue
                        else:
                            # Points are concyclic
                            return False
        return True

    def check_valid(D, pts):
        """Full validity check: rational distances, no 3 collinear, no 4 concyclic."""
        n = len(pts)
        # Check distances
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x1 - x2
                dy = y1 - y2
                d2 = dx * dx + D * dy * dy
                if not is_rational_square(d2):
                    return False
        # Check collinearity
        if not check_collinear(pts):
            return False
        # Check concyclicity
        if not check_concyclic(D, pts):
            return False
        return True

    # Known 5-point configuration with D=1
    fallback_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    fallback_result = {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in fallback_points]}

    def try_find_set(D, max_points=6, time_limit=None):
        """Try to find a set of max_points with given D using a greedy search."""
        if time_limit is None:
            time_limit = deadline
        
        # Start with P1=(0,0), P2=(1,0)
        P1 = (Fraction(0), Fraction(0))
        P2 = (Fraction(1), Fraction(0))
        pts = [P1, P2]
        
        # Search for P3
        found_p3 = False
        for q1_num in range(1, 100):
            for q2_num in range(1, 100):
                if time.time() > time_limit:
                    return None
                q1 = Fraction(q1_num, 1)
                q2 = Fraction(q2_num, 1)
                # x3 = (q1^2 - q2^2 + 1) / 2
                x3 = (q1 * q1 - q2 * q2 + 1) / 2
                # D * y3^2 = q1^2 - x3^2
                rhs = q1 * q1 - x3 * x3
                if rhs < 0:
                    continue
                y3_sq = rhs / D
                if not is_rational_square(y3_sq):
                    continue
                y3 = sqrt_frac(y3_sq)
                if y3 is None:
                    continue
                
                P3 = (x3, y3)
                # Check distinct
                if P3 == P1 or P3 == P2:
                    continue
                # Check collinear (P1, P2, P3)
                if (P2[0] - P1[0]) * (P3[1] - P1[1]) - (P2[1] - P1[1]) * (P3[0] - P1[0]) == 0:
                    continue
                
                pts.append(P3)
                found_p3 = True
                break
            if found_p3:
                break
        
        if not found_p3:
            return None
        
        # Search for P4, P5, P6
        for target_idx in range(3, max_points):
            found = False
            # Iterate distances to P1, P2, P3
            for q1_num in range(1, 80):
                for q2_num in range(1, 80):
                    for q3_num in range(1, 80):
                        if time.time() > time_limit:
                            return None
                        q1 = Fraction(q1_num, 1)
                        q2 = Fraction(q2_num, 1)
                        q3 = Fraction(q3_num, 1)
                        
                        # Solve for x, y using radical axes
                        x1, y1 = pts[0]
                        x3, y3 = pts[2]
                        A1 = 2 * (x3 - x1)
                        B1 = 2 * D * (y3 - y1)
                        C1 = x1 * x1 + D * y1 * y1 - x3 * x3 - D * y3 * y3 + q3 * q3 - q1 * q1
                        
                        x2, y2 = pts[1]
                        A2 = 2 * (x3 - x2)
                        B2 = 2 * D * (y3 - y2)
                        C2 = x2 * x2 + D * y2 * y2 - x3 * x3 - D * y3 * y3 + q3 * q3 - q2 * q2
                        
                        # Solve A1 x + B1 y = C1
                        # A2 x + B2 y = C2
                        det = A1 * B2 - A2 * B1
                        if det == 0:
                            continue
                        
                        x_new = (C1 * B2 - C2 * B1) / det
                        y_new = (A1 * C2 - A2 * C1) / det
                        
                        P_new = (x_new, y_new)
                        
                        # Check if distinct and not collinear with any pair
                        valid = True
                        for i in range(len(pts)):
                            if P_new == pts[i]:
                                valid = False
                                break
                            for j in range(i + 1, len(pts)):
                                xi, yi = pts[i]
                                xj, yj = pts[j]
                                if (xj - xi) * (y_new - yi) - (yj - yi) * (x_new - xi) == 0:
                                    valid = False
                                    break
                            if not valid:
                                break
                        
                        if not valid:
                            continue
                        
                        # Check distances to all existing points
                        ok = True
                        for k in range(len(pts)):
                            xk, yk = pts[k]
                            dx = x_new - xk
                            dy = y_new - yk
                            d2 = dx * dx + D * dy * dy
                            if not is_rational_square(d2):
                                ok = False
                                break
                        if not ok:
                            continue
                        
                        pts.append(P_new)
                        found = True
                        break
                    if found:
                        break
                if found:
                    break
            
            if not found:
                return None
        
        return pts

    # Try different D values
    best_result = fallback_result
    best_count = 5
    
    for D in [1, 2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 46, 47, 51, 53, 55, 57, 58, 59, 61, 62, 65, 66, 67, 69, 70, 71, 73, 74, 77, 78, 79, 82, 83, 85, 86, 87, 89, 90, 93, 94, 95, 97]:
        if time.time() > deadline - 5:
            break
        pts = try_find_set(D, max_points=6, time_limit=deadline - 10)
        if pts:
            # Verify concyclicity for the final set
            if check_valid(D, pts):
                result = {'D': D, 'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in pts]}
                if len(pts) > best_count:
                    best_count = len(pts)
                    best_result = result
                if best_count >= 6:
                    return best_result
    
    return best_result

# EVOLVE-BLOCK-END
