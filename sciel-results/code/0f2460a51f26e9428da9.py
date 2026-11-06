"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time
from fractions import Fraction
from itertools import combinations

def is_rational_sqrt(n):
    """Check if n is a perfect square of a rational number.
    Here n is an integer (from integer coords) or Fraction.
    If n is int, check if it's a perfect square int.
    If n is Fraction, check if num and den are perfect squares."""
    if isinstance(n, Fraction):
        num = n.numerator
        den = n.denominator
        if num < 0 or den < 0:
            return False
        r_num = math.isqrt(num)
        r_den = math.isqrt(den)
        return r_num * r_num == num and r_den * r_den == den
    else:
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

def dist_sq_rational(p, q, D):
    """Calculate squared distance between (x1, y1) and (x2, y2) in Q x Q*sqrt(D).
    p, q are tuples of Fractions (x, y).
    Returns True if distance is rational (squared dist is perfect square of rational)."""
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    d2 = dx*dx + D*dy*dy
    return is_rational_sqrt(d2)

def check_no_3_collinear(points):
    """Check no three points are collinear. Points are (x, y) Fractions."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                # Cross product of (P2-P1) and (P3-P1)
                cross = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
                if cross == 0:
                    return False
    return True

def check_no_4_concyclic(points, D):
    """Check no four points are concyclic.
    Points are (x, y) Fractions, in space Q x Q*sqrt(D).
    Concyclicity in this metric is equivalent to lying on a circle in the transformed space?
    Actually, the condition "no four concyclic" usually refers to the standard Euclidean geometry of the points in the plane.
    The points are (x, y*sqrt(D)). Let Y = y*sqrt(D).
    We need to check if 4 points (x_i, Y_i) lie on a circle in the standard plane.
    Circle equation: x^2 + Y^2 + Ax + BY + C = 0.
    Since Y = y*sqrt(D), Y^2 = D*y^2.
    So: x^2 + D*y^2 + Ax + B(y*sqrt(D)) + C = 0.
    This involves sqrt(D). For this to hold for rational x,y, the coefficient of sqrt(D) must be 0?
    No, the points are fixed. We just check if they lie on *some* circle.
    A circle in the plane is defined by 3 points. The 4th lies on it iff the determinant is 0.
    Determinant of:
    | x  Y  x^2+Y^2  1 |
    | x1 Y1 x1^2+Y1^2 1 |
    ...
    Since Y = y*sqrt(D), the terms will involve sqrt(D).
    If D is not a square, the determinant will be of the form A + B*sqrt(D).
    For the points to be concyclic, we need A=0 and B=0.
    """
    n = len(points)
    if n < 4:
        return True
    
    # We need to check all 4-point subsets.
    # To be efficient, we can use the property that if 4 points are concyclic,
    # the circle through any 3 must contain the 4th.
    
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    p1 = points[i]
                    p2 = points[j]
                    p3 = points[k]
                    p4 = points[l]
                    
                    x1, y1 = p1
                    x2, y2 = p2
                    x3, y3 = p3
                    x4, y4 = p4
                    
                    # Y values are y*sqrt(D). We work in the field Q(sqrt(D)).
                    # Let Y1 = y1*sqrt(D).
                    # Circle through P1, P2, P3.
                    # Equation: x^2 + Y^2 + Ax + BY + C = 0
                    # Substitute Y: x^2 + D*y^2 + Ax + B*y*sqrt(D) + C = 0
                    # This must hold for (x1, y1), (x2, y2), (x3, y3).
                    # Let's separate rational and irrational parts.
                    # x_i^2 + D*y_i^2 + C + A*x_i + B*y_i*sqrt(D) = 0
                    # Rational part: x_i^2 + D*y_i^2 + C + A*x_i = 0
                    # Irrational part: B*y_i = 0  => If any y_i != 0, then B=0.
                    
                    # If B=0, the equation is x^2 + D*y^2 + Ax + C = 0.
                    # This is an ellipse (or circle if D=1).
                    # If B!=0, then all y_i must be 0? No, if B!=0, then y_i must be 0 for the irrational part to vanish?
                    # Wait, the equation is an identity for the points.
                    # If B != 0, then y1=y2=y3=0. But we check no 3 collinear, so if y1=y2=y3=0, they are collinear on x-axis.
                    # So, for non-collinear points, we must have B=0.
                    # Thus, the "circle" in the metric space corresponds to a conic section in the (x,y) plane:
                    # x^2 + D*y^2 + Ax + C = 0.
                    
                    # So we check if 4 points lie on a conic of the form x^2 + D*y^2 + Ax + C = 0.
                    # This conic is determined by 3 points (solving for A, C).
                    # Let's solve for A, C using P1, P2, P3.
                    # x1^2 + D*y1^2 + A*x1 + C = 0  (1)
                    # x2^2 + D*y2^2 + A*x2 + C = 0  (2)
                    # x3^2 + D*y3^2 + A*x3 + C = 0  (3)
                    
                    # (1)-(2): A(x1-x2) + (x1^2-x2^2 + D(y1^2-y2^2)) = 0
                    # A = - (x1^2-x2^2 + D(y1^2-y2^2)) / (x1-x2)
                    # Note: x1 != x2 or y1 != y2. If x1==x2 and y1==y2, points are same (distinctness checked).
                    # If x1==x2, then y1!=y2 (else same point). Then denom is 0.
                    # In that case, we use a different pair or handle separately.
                    
                    # Let's use a determinant test for the conic.
                    # Points (x,y) lie on x^2 + D*y^2 + Ax + C = 0 iff:
                    # det(
                    #   [1, x, x^2+D*y^2],
                    #   [1, x1, x1^2+D*y1^2],
                    #   [1, x2, x2^2+D*y2^2],
                    #   [1, x3, x3^2+D*y3^2]
                    # ) = 0 ? No, that's for 4 points to determine A, C?
                    # We have 2 unknowns A, C. So 3 points determine A, C uniquely (if not degenerate).
                    # The 4th point must satisfy the equation.
                    
                    # Calculate A, C from P1, P2, P3.
                    # If x1 == x2:
                    #   Use P1, P3, P2?
                    #   Actually, if x1==x2, then y1!=y2.
                    #   Eq 1: x1^2 + D*y1^2 + A*x1 + C = 0
                    #   Eq 2: x1^2 + D*y2^2 + A*x1 + C = 0
                    #   Subtract: D(y1^2 - y2^2) = 0. Since D!=0, y1^2=y2^2.
                    #   y1 != y2 => y1 = -y2.
                    #   This is a specific case.
                    
                    # General solution using linear algebra on [x, 1] vector.
                    # A*x1 + C = -(x1^2 + D*y1^2)
                    # A*x2 + C = -(x2^2 + D*y2^2)
                    # A*x3 + C = -(x3^2 + D*y3^2)
                    
                    # Let V1 = [x1, 1], V2 = [x2, 1], V3 = [x3, 1].
                    # Let R1 = -(x1^2 + D*y1^2), etc.
                    # We need A, C such that V1 dot [A,C] = R1, etc.
                    # If x1 != x2:
                    #   A = (R1 - R2) / (x1 - x2)
                    #   C = R1 - A*x1
                    #   Check if A*x3 + C == R3.
                    #   If yes, check P4.
                    #   If x1 == x2:
                    #   Then we must use P1, P3 and P2, P3.
                    #   If x1==x2 and x1!=x3:
                    #   A = (R1 - R3) / (x1 - x3)
                    #   C = R1 - A*x1
                    #   Check P2 and P4.
                    #   If x1==x2==x3: All 3 have same x.
                    #   Then R1 must equal R2 must equal R3 for a solution to exist?
                    #   R1 = -(x^2 + D*y1^2). R2 = -(x^2 + D*y2^2).
                    #   R1=R2 => y1^2=y2^2.
                    #   If y1 != y2, y1=-y2.
                    #   Then the conic is x^2 + D*y^2 + Ax + C = 0.
                    #   This is a vertical symmetry case.
                    #   We can just check if P4 satisfies the condition derived from P1, P2.
                    #   Actually, if 3 points have same x, they are on a vertical line.
                    #   Wait, "no three collinear". If x1=x2=x3, they are collinear!
                    #   So this case is already excluded by `check_no_3_collinear`.
                    
                    # So, at least 2 of the 3 points must have different x.
                    # We can always pick two with different x.
                    
                    # Let's pick p_i, p_j with x_i != x_j.
                    # Compute A, C.
                    # Verify p_k.
                    # If p_k fails, they are not on this conic.
                    # If p_k passes, check p_l.
                    
                    # To be safe, we just iterate pairs to find a valid A, C.
                    found_valid_conic = False
                    for i2 in range(3):
                        for j2 in range(i2+1, 3):
                            pi = [p1, p2, p3][i2]
                            pj = [p1, p2, p3][j2]
                            xi, yi = pi
                            xj, yj = pj
                            if xi == xj:
                                continue
                            Ri = -(xi*xi + D*yi*yi)
                            Rj = -(xj*xj + D*yj*yj)
                            A = (Ri - Rj) / (xi - xj)
                            C = Ri - A*xi
                            # Check 3rd point
                            pk = [p1, p2, p3][3 - i2 - j2] # index of the remaining one
                            xk, yk = pk
                            Rk = -(xk*xk + D*yk*yk)
                            if A*xk + C != Rk:
                                continue
                            # Check 4th point
                            xl, yl = p4
                            Rl = -(xl*xl + D*yl*yl)
                            if A*xl + C == Rl:
                                found_valid_conic = True
                                break
                        if found_valid_conic:
                            break
                    if found_valid_conic:
                        return False
    return True

def verify_set(points, D):
    """Full verification."""
    n = len(points)
    if n < 3:
        return False
    
    # Distinctness
    for i in range(n):
        for j in range(i+1, n):
            if points[i] == points[j]:
                return False
            
    # Rational distances
    for i in range(n):
        for j in range(i+1, n):
            if not dist_sq_rational(points[i], points[j], D):
                return False
                
    # No 3 collinear
    if not check_no_3_collinear(points):
        return False
        
    # No 4 concyclic
    if not check_no_4_concyclic(points, D):
        return False
        
    return True

def construct():
    t0 = time.time()
    deadline = 115.0
    
    # Known 5-point set (D=1)
    # Points: (0,0), (-56,33), (-16,63), (0,33), (16,63)
    pts5_int = [(0,0), (-56,33), (-16,63), (0,33), (16,63)]
    pts5 = [(Fraction(x), Fraction(y)) for x,y in pts5_int]
    D = 1
    
    # Verify the 5-point set
    if verify_set(pts5, D):
        best_points = pts5
        best_D = D
    else:
        # Fallback: simple 3 points
        best_points = [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(0)), (Fraction(0), Fraction(4))]
        best_D = 1

    # Try to find a 6-point set
    # Strategy: Random search for 6 points with D=1 in a box
    # This is expensive, so we limit it.
    
    # If we have time, try to extend the 5-point set or find a new one.
    # Given the difficulty, we might not find one.
    # But we must try.
    
    # Try a small search for 6 points with D=1
    # We'll use a greedy approach with random restarts.
    
    max_search_time = 100.0
    
    def try_find_6_points():
        t_start = time.time()
        for _ in range(200): # 200 restarts
            if time.time() - t_start > max_search_time:
                break
            
            # Randomly pick 6 points from a grid?
            # Or generate random integer points?
            # Let's try to build a set greedily.
            
            pts = []
            # Start with (0,0)
            pts.append((Fraction(0), Fraction(0)))
            
            # Add 5 more
            for _ in range(5):
                found = False
                # Try random points in a box
                for _ in range(500):
                    x = random.randint(-100, 100)
                    y = random.randint(-100, 100)
                    if x == 0 and y == 0:
                        continue
                    new_pt = (Fraction(x), Fraction(y))
                    # Check distance to all existing points
                    valid = True
                    for p in pts:
                        if not dist_sq_rational(p, new_pt, 1):
                            valid = False
                            break
                    if not valid:
                        continue
                    # Check collinearity with any 2 existing points
                    if len(pts) >= 2:
                        collinear = False
                        for i in range(len(pts)):
                            for j in range(i+1, len(pts)):
                                p1 = pts[i]
                                p2 = pts[j]
                                cross = (p2[0]-p1[0])*(new_pt[1]-p1[1]) - (new_pt[0]-p1[0])*(p2[1]-p1[1])
                                if cross == 0:
                                    collinear = True
                                    break
                            if collinear:
                                break
                        if collinear:
                            continue
                    # Check concyclicity with any 3 existing points
                    if len(pts) >= 3:
                        concyclic = False
                        # Check if new_pt lies on conic of any 3 existing points
                        for i in range(len(pts)):
                            for j in range(i+1, len(pts)):
                                for k in range(j+1, len(pts)):
                                    p1, p2, p3 = pts[i], pts[j], pts[k]
                                    # Check if p1,p2,p3,new_pt are concyclic
                                    # Use the logic from check_no_4_concyclic
                                    # We can just add new_pt to the set and call check_no_4_concyclic?
                                    # That's O(n^4). For n=6, it's small.
                                    # But we are in a loop, so it's expensive.
                                    # Let's just do the full check on the candidate set of size len(pts)+1.
                                    candidate = pts + [new_pt]
                                    if not check_no_4_concyclic(candidate, 1):
                                        concyclic = True
                                        break
                                if concyclic:
                                    break
                            if concyclic:
                                break
                        if concyclic:
                            continue
                    
                    # If we get here, it's a valid addition
                    pts.append(new_pt)
                    found = True
                    break
                if not found:
                    break
            
            if len(pts) == 6:
                # Full verification
                if verify_set(pts, 1):
                    return pts, 1
        return None

    result = try_find_6_points()
    if result:
        best_points, best_D = result

    # Convert back to required format
    return {'D': best_D, 'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in best_points]}
# EVOLVE-BLOCK-END
