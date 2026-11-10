import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Helper to check if a set of points is valid
    def is_valid(points, D):
        n = len(points)
        if n < 3:
            return False
        
        # Check distinctness
        for i in range(n):
            for j in range(i+1, n):
                if points[i][0] == points[j][0] and points[i][1] == points[j][1]:
                    return False
        
        # Check rational distances
        for i in range(n):
            for j in range(i+1, n):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                dist_sq = dx*dx + D*dy*dy
                # dist_sq must be a square of a rational
                # Since dx, dy are Fraction, dist_sq is Fraction
                num, den = dist_sq.numerator, dist_sq.denominator
                # Check if num/den is a perfect square
                sqrt_num = int(math.isqrt(num))
                sqrt_den = int(math.isqrt(den))
                if sqrt_num * sqrt_num != num or sqrt_den * sqrt_den != den:
                    return False
                # Also check that the square root is rational (implied by above for positive dist_sq)
                # But we need to ensure dist_sq >= 0
                if dist_sq < 0:
                    return False
        
        # Check no three collinear
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    x1, y1 = points[i][0], points[i][1]
                    x2, y2 = points[j][0], points[j][1]
                    x3, y3 = points[k][0], points[k][1]
                    # Determinant of matrix with rows (x1, y1, 1), etc.
                    # (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1) != 0
                    if (x2-x1)*(y3-y1) == (x3-x1)*(y2-y1):
                        return False
        
        # Check no four concyclic
        # Equation of circle: x^2 + y^2 + Ax + By + C = 0
        # For point (x_i, y_i): x_i^2 + y_i^2 + A x_i + B y_i + C = 0
        # We can solve for A, B, C using 3 points and check the 4th.
        # However, in Q(sqrt(D)), the circle equation is slightly different.
        # Distance squared from (x,y) to (x_i, y_i) is (x-x_i)^2 + D(y-y_i)^2.
        # This is not a standard Euclidean circle.
        # "Concyclic" in the context of rational distances usually refers to the Euclidean geometry of the embedded space.
        # The task says "no four concyclic". This implies the standard Euclidean circle in the (x, y*sqrt(D)) plane.
        # The equation is x^2 + D y^2 + Ax + By + C = 0.
        
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        x1, y1 = points[i][0], points[i][1]
                        x2, y2 = points[j][0], points[j][1]
                        x3, y3 = points[k][0], points[k][1]
                        x4, y4 = points[l][0], points[l][1]
                        
                        # System:
                        # x1^2 + D y1^2 + A x1 + B y1 + C = 0
                        # x2^2 + D y2^2 + A x2 + B y2 + C = 0
                        # x3^2 + D y3^2 + A x3 + B y3 + C = 0
                        # Solve for A, B, C.
                        
                        # Use Cramer's rule or simple substitution.
                        # Let's use Fraction arithmetic.
                        
                        # Subtract 1 from 2:
                        # (x2^2 - x1^2) + D(y2^2 - y1^2) + A(x2-x1) + B(y2-y1) = 0
                        # Subtract 1 from 3:
                        # (x3^2 - x1^2) + D(y3^2 - y1^2) + A(x3-x1) + B(y3-y1) = 0
                        
                        A1 = (x2-x1)
                        B1 = (y2-y1)
                        C1 = -((x2^2 - x1^2) + D*(y2^2 - y1^2))
                        
                        A2 = (x3-x1)
                        B2 = (y3-y1)
                        C2 = -((x3^2 - x1^2) + D*(y3^2 - y1^2))
                        
                        det = A1*B2 - A2*B1
                        if det == 0:
                            continue # Points collinear, already handled, but skip circle check
                        
                        A_val = (C1*B2 - C2*B1) / det
                        B_val = (A1*C2 - A2*C1) / det
                        # C_val = -x1^2 - D*y1^2 - A_val*x1 - B_val*y1
                        
                        # Check if point 4 lies on this circle
                        # x4^2 + D y4^2 + A_val x4 + B_val y4 + C_val = 0
                        # C_val = -x1^2 - D*y1^2 - A_val*x1 - B_val*y1
                        # LHS = x4^2 + D y4^2 + A_val x4 + B_val y4 - x1^2 - D*y1^2 - A_val*x1 - B_val*y1
                        lhs = (x4^2 - x1^2) + D*(y4^2 - y1^2) + A_val*(x4-x1) + B_val*(y4-y1)
                        if lhs == 0:
                            return False
        
        return True

    # Strategy: Start with the seed points and try to extend.
    # Seed: 5 points
    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    D = 1
    
    best_points = seed_points
    
    # Verify seed is valid (should be)
    if is_valid(seed_points, D):
        best_points = seed_points
    
    # Try to find a 6th point
    # We will search for (x, y) with small integer coordinates
    # Given the spread of seed points, we might need a wider range or different D.
    # But D=1 is simplest.
    
    # If we can't find a 6th point for D=1 quickly, we might try other D or different base sets.
    # However, the seed is already valid.
    
    # Let's try to add a point to the seed.
    # The seed points are:
    # P0: (0,0)
    # P1: (-56, 33)
    # P2: (-16, 63)
    # P3: (0, 33)
    # P4: (16, 63)
    
    # Let's try to find a point (x,y) such that dist to all 5 is rational.
    # This is a system of 5 equations.
    # We can iterate x, y in a range.
    
    found_6 = False
    # Range search
    # The points are roughly in x: [-56, 16], y: [0, 63].
    # Let's search x in [-100, 100], y in [-100, 100]
    
    # To speed up, we can use the fact that dist^2 must be a square.
    # d0^2 = x^2 + y^2 = k0^2
    # d1^2 = (x+56)^2 + (y-33)^2 = k1^2
    # ...
    
    # This is expensive. Let's try a different base set that is easier to extend.
    # The "Heronian" triangle based sets are often easier.
    
    # Let's try the set: (0,0), (1,0), (0,1) and try to add points.
    # We already analyzed this.
    
    # Let's try a known 6-point set if we can find one in literature or derive it.
    # One known 6-point set:
    # (0,0), (1,0), (0,1), (4/3, 0), (0, 3/4), (4/3, 3/4) ?
    # Let's check P6 = (4/3, 3/4) against P1(1,0), P2(0,1), P3(4/3,0), P4(0,3/4).
    # d(P6, P1)^2 = (4/3 - 1)^2 + (3/4 - 0)^2 = 1/9 + 9/16 = 16/144 + 81/144 = 97/144. Not a square.
    
    # So that set is not valid.
    
    # Let's stick to the seed and try to find a 6th point with a bounded search.
    # If we fail, we return 5.
    
    # To improve odds, we can try to construct a 6-point set from scratch using the parametrization.
    # P1=(0,0), P2=(1,0).
    # P3=(x3, y3) such that d(P1,P3), d(P2,P3) rational.
    # P4=(x4, y4) such that d(Pi,P4) rational for i=1,2,3.
    # P5, P6 similarly.
    
    # This is complex. Given the time constraint, returning the valid 5-point set is a safe fallback.
    # But we want to beat 1.0.
    
    # Let's try a very specific 6-point configuration found in literature.
    # "Integral point sets in general position"
    # A known 6-point set:
    # (0,0), (1,0), (0,1), (3/4, 0), (0, 4/3), (3/4, 4/3) ?
    # Let's check P6=(3/4, 4/3).
    # d(P6, P1)^2 = 9/16 + 16/9 = (81+256)/144 = 337/144. Not square.
    
    # It seems hard to guess.
    
    # We will return the 5-point set.
    # Wait, the task says "8 points would be new". This implies 6 and 7 are known.
    # The seed is 5.
    # If we can't find 6, we are stuck at 1.0.
    
    # Let's try to search for a 6th point for the seed more efficiently.
    # We can fix y and solve for x?
    # No, just brute force small integers.
    
    # If we don't find a 6th point for D=1, we might try D=2, 3, etc.
    # But changing D requires re-validating the seed or finding a new set.
    
    # Given the risk, I will return the seed 5-point set.
    # However, I will add a quick check for a 6th point in a small neighborhood of the seed.
    
    for x in range(-100, 101):
        for y in range(-100, 101):
            if time.time() > deadline:
                break
            new_point = (Fraction(x), Fraction(y))
            candidate = seed_points + [new_point]
            if is_valid(candidate, D):
                best_points = candidate
                found_6 = True
                break
        if found_6 or time.time() > deadline:
            break
    
    if found_6:
        return {'D': D, 'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in best_points]}
    else:
        return {'D': D, 'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in best_points]}
