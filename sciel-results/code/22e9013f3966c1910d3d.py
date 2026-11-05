"""
Search for integral point sets in general position (no three collinear, no four concyclic).
We search over D in a small set, and over integer points (or rational with small denominators).
All pairwise squared distances must be D times a rational square (i.e., distance is rational).
"""

from fractions import Fraction
import math
import time

def is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def squarefree_part(n):
    """Return squarefree part of n (for n > 0)."""
    if n == 0:
        return 1
    sf = 1
    i = 2
    while i * i <= n:
        while n % (i * i) == 0:
            n //= (i * i)
            sf *= i * i
        i += 1
    return n * sf  # Wait, this is wrong. Let me fix.

def squarefree_part(n):
    """Return the squarefree part of positive integer n."""
    if n == 0:
        return 1
    sf = 1
    n_abs = abs(n)
    i = 2
    while i * i <= n_abs:
        while n_abs % (i * i) == 0:
            n_abs //= (i * i)
            sf *= i * i
        i += 1
    return n_abs  # The squarefree part is n_abs after removing all square factors

def gcd3(a, b, c):
    from math import gcd
    return gcd(gcd(a, b), c)

def is_square_perfect(num, den):
    """Check if num/den is a perfect square of a rational (num, den positive)."""
    from math import isqrt
    if num < 0 or den <= 0:
        return False
    # Check if num and den are perfect squares
    r_num = isqrt(num)
    r_den = isqrt(den)
    return r_num * r_num == num and r_den * r_den == den

def dist_sq_rational(p1, p2, D):
    """
    p1, p2: (x_num, x_den, y_num, y_den) representing (x, y*sqrt(D))
    Returns (num, den) for the squared distance, or None if not rational.
    """
    x1, y1 = Fraction(p1[0], p1[1]), Fraction(p1[2], p1[3])
    x2, y2 = Fraction(p2[0], p2[1]), Fraction(p2[2], p2[3])
    dx = x1 - x2
    dy = y1 - y2
    # dist^2 = dx^2 + D * dy^2
    dsq = dx * dx + D * dy * dy
    return dsq

def check_all_dist_rational(points, D):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            dsq = dist_sq_rational(points[i], points[j], D)
            # dsq is a Fraction, check if it's a perfect square
            if dsq < 0:
                return False
            num = dsq.numerator
            den = dsq.denominator
            if not is_square_perfect(num, den):
                return False
    return True

def no_three_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Check if points i, j, k are collinear
                # (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i) == 0
                x_i = Fraction(points[i][0], points[i][1])
                y_i = Fraction(points[i][2], points[i][3])
                x_j = Fraction(points[j][0], points[j][1])
                y_j = Fraction(points[j][2], points[j][3])
                x_k = Fraction(points[k][0], points[k][1])
                y_k = Fraction(points[k][2], points[k][3])
                det = (x_j - x_i) * (y_k - y_i) - (y_j - y_i) * (x_k - x_i)
                if det == 0:
                    return False
    return True

def no_four_concyclic(points, D):
    """
    Check that no four points are concyclic.
    Four points (x_i, y_i*sqrt(D)) are concyclic if and only if
    the determinant of the circle equation is zero.
    Circle: x^2 + y^2 + a*x + b*y*sqrt(D) + c = 0
    For point i: x_i^2 + D*y_i^2 + a*x_i + b*sqrt(D)*y_i + c = 0
    This is linear in a, b, c. For 4 points, the system is overdetermined.
    We check if the 4x4 determinant is zero.
    """
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    # Points i, j, k, l
                    # Each gives: x^2 + D*y^2 + a*x + b*sqrt(D)*y + c = 0
                    # => a*x + b*sqrt(D)*y + c = -(x^2 + D*y^2)
                    # For 4 points, we have 4 equations in 3 unknowns (a, b, c)
                    # The system has a solution iff the 4x4 augmented matrix has rank <= 3
                    # Equivalently, the determinant of the 4x4 matrix is 0.
                    pts = [points[i], points[j], points[k], points[l]]
                    det = 0
                    # Build 4x4 matrix
                    M = []
                    for p in pts:
                        x = Fraction(p[0], p[1])
                        y = Fraction(p[2], p[3])
                        # Row: [x, y, 1, x^2 + D*y^2]
                        # But the equation is a*x + b*sqrt(D)*y + c = -(x^2 + D*y^2)
                        # So the augmented matrix row is [x, sqrt(D)*y, 1, -(x^2 + D*y^2)]
                        # But sqrt(D) is irrational, so we can't put it in a rational matrix.
                        # Instead, we can write the system as:
                        # a*x_i + b*y_i + c = -(x_i^2 + D*y_i^2) / sqrt(D) ? No.
                        # Let me reconsider.
                        # The circle equation in the (x, Y) plane where Y = y*sqrt(D):
                        # x^2 + Y^2 + a*x + b*Y + c = 0
                        # So for point i: x_i^2 + (y_i*sqrt(D))^2 + a*x_i + b*(y_i*sqrt(D)) + c = 0
                        # => x_i^2 + D*y_i^2 + a*x_i + b*sqrt(D)*y_i + c = 0
                        # => a*x_i + b*sqrt(D)*y_i + c = -(x_i^2 + D*y_i^2)
                        # This is 3 equations (for 3 points) in 3 unknowns (a, b, c).
                        # For 4 points, we need consistency.
                        # We can eliminate a and c:
                        # a = (-(x_i^2 + D*y_i^2) - b*sqrt(D)*y_i - c) / x_i
                        # This is getting messy. Let me use a different approach.
                        #
                        # Alternative: 4 points are concyclic iff the determinant of the following 4x4 matrix is 0:
                        # | x1^2 + D*y1^2, x1, y1, 1 |
                        # | x2^2 + D*y2^2, x2, y2, 1 |
                        # | x3^2 + D*y3^2, x3, y3, 1 |
                        # | x4^2 + D*y4^2, x4, y4, 1 |
                        # This is because the circle equation is x^2 + Y^2 + a*x + b*Y + c = 0,
                        # where Y = y*sqrt(D). So x^2 + Y^2 = x^2 + D*y^2.
                        # The system is:
                        # x1^2 + D*y1^2 + a*x1 + b*y1*sqrt(D) + c = 0
                        # But b*y1*sqrt(D) is not rational. So we can't directly use this.
                        #
                        # Let me think again. The points are (x_i, Y_i) where Y_i = y_i*sqrt(D).
                        # They are concyclic if there exist a, b, c such that
                        # x_i^2 + Y_i^2 + a*x_i + b*Y_i + c = 0 for all i.
                        # This is a system of linear equations in a, b, c.
                        # For 3 points, we can solve for a, b, c. For the 4th point, we check if it satisfies.
                        # But b*Y_i = b*y_i*sqrt(D), so the equations are:
                        # a*x_i + b*(y_i*sqrt(D)) + c = -(x_i^2 + D*y_i^2)
                        # Let's separate rational and irrational parts:
                        # a*x_i + c + b*sqrt(D)*y_i = -(x_i^2 + D*y_i^2)
                        # The LHS has a rational part (a*x_i + c) and an irrational part (b*sqrt(D)*y_i).
                        # The RHS is rational. So we need:
                        # a*x_i + c = -(x_i^2 + D*y_i^2)
                        # b*y_i = 0
                        # This is only possible if b=0 (for all i with y_i != 0) or all y_i = 0.
                        # So for generic points, the only circles are of the form x^2 + Y^2 + a*x + c = 0 (b=0).
                        # Wait, this is not right. The issue is that sqrt(D) is irrational, so the coefficients a, b, c
                        # must be such that the equation holds. But a, b, c are real numbers, not necessarily rational.
                        # So the system is:
                        # a*x_i + b*(y_i*sqrt(D)) + c = -(x_i^2 + D*y_i^2)
                        # This is a linear system in a, b, c with real coefficients.
                        # For 4 points, we have 4 equations in 3 unknowns. The system is consistent iff
                        # the 4x4 determinant of the augmented matrix is 0.
                        # The augmented matrix is:
                        # [ x1, y1*sqrt(D), 1, -(x1^2 + D*y1^2) ]
                        # [ x2, y2*sqrt(D), 1, -(x2^2 + D*y2^2) ]
                        # [ x3, y3*sqrt(D), 1, -(x3^2 + D*y3^2) ]
                        # [ x4, y4*sqrt(D), 1, -(x4^2 + D*y4^2) ]
                        # The determinant is a polynomial in sqrt(D). For it to be 0, both the rational and irrational
                        # parts must be 0.
                        # Let's write the determinant as A + B*sqrt(D) where A, B are rational.
                        # We need A = 0 and B = 0.
                        #
                        # This is complex. Let me use a simpler check: for 3 points, compute the circle, then check the 4th.
                        pass
    return True

def no_four_concyclic_v2(points, D):
    """
    Check no four concyclic by computing the circle through 3 points and checking the 4th.
    """
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Compute circle through points i, j, k
                x_i = Fraction(points[i][0], points[i][1])
                y_i = Fraction(points[i][2], points[i][3])
                Y_i = y_i * math.sqrt(D)  # This is irrational, so we can't use it directly.
                # Instead, let's use the fact that the circle equation is:
                # x^2 + Y^2 + a*x + b*Y + c = 0
                # For point i: x_i^2 + D*y_i^2 + a*x_i + b*sqrt(D)*y_i + c = 0
                # Let's separate: (a*x_i + c + x_i^2 + D*y_i^2) + b*sqrt(D)*y_i = 0
                # Since sqrt(D) is irrational (D is squarefree > 1), we need:
                # a*x_i + c + x_i^2 + D*y_i^2 = 0
                # b*y_i = 0
                # So if y_i != 0, then b = 0. If all y_i = 0, then the points are on the x-axis.
                # So for D > 1, the circle must have b = 0 (if any y_i != 0).
                # This means the circle equation is x^2 + Y^2 + a*x + c = 0.
                # Let's assume b = 0 for now.
                # Then we have:
                # a*x_i + c = -(x_i^2 + D*y_i^2)
                # This is a linear system in a and c.
                # For 2 points, we can solve. For the 3rd, we check.
                # Actually, for 2 points, we have 2 equations in 2 unknowns, so we can solve.
                # Then check the 3rd.
                #
                # Let's use points i and j to solve for a and c:
                # a*x_i + c = -(x_i^2 + D*y_i^2)
                # a*x_j + c = -(x_j^2 + D*y_j^2)
                # Subtract: a*(x_i - x_j) = -(x_i^2 + D*y_i^2) + (x_j^2 + D*y_j^2)
                # a = (x_j^2 + D*y_j^2 - x_i^2 - D*y_i^2) / (x_i - x_j)  [if x_i != x_j]
                if x_i == x_j:
                    continue
                a_num = (x_j * x_j + D * y_j * y_j) - (x_i * x_i + D * y_i * y_i)
                a_den = x_i - x_j
                a = a_num / a_den
                c = -(x_i * x_i + D * y_i * y_i) - a * x_i
                # Check point k
                if a * x_k + c != -(x_k * x_k + D * y_k * y_k):
                    continue
                # The circle through i, j, k is x^2 + Y^2 + a*x + c = 0
                # Now check all other points l
                for l in range(n):
                    if l == i or l == j or l == k:
                        continue
                    x_l = Fraction(points[l][0], points[l][1])
                    y_l = Fraction(points[l][2], points[l][3])
                    if a * x_l + c == -(x_l * x_l + D * y_l * y_l):
                        # Point l is on the circle, so 4 concyclic
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    # Try D = 1 first (integer points)
    # We'll search for integer points with rational pairwise distances.
    # This means squared distances must be perfect squares.
    
    # Start with the known 5-point set
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    D = 1
    
    # Try to add more points
    best_points = base_points[:]
    
    # Try to add a 6th point
    # Search over a grid of integer points
    for x in range(-200, 201):
        for y in range(-200, 201):
            if time.time() > deadline - 30:
                return {'D': D, 'points': [[[x[0], 1], [x[1], 1]] for x in best_points]}
            
            new_point = (x, y)
            # Check if it's distinct from existing points
            if new_point in best_points:
                continue
            
            # Check all pairwise distances
            all_ok = True
            for p in best_points:
                dx = x - p[0]
                dy = y - p[1]
                dsq = dx * dx + dy * dy
                if not is_square(dsq):
                    all_ok = False
                    break
            if not all_ok:
                continue
            
            # Check no three collinear
            candidate = best_points + [new_point]
            if not no_three_collinear(candidate):
                continue
            
            # Check no four concyclic
            if not no_four_concyclic_v2(candidate, D):
                continue
            
            # Found a valid 6-point set!
            best_points = candidate
            break
        if len(best_points) > 5:
            break
    
    # Try to add a 7th point if we have 6
    if len(best_points) >= 6:
        for x in range(-300, 301):
            for y in range(-300, 301):
                if time.time() > deadline - 10:
                    return {'D': D, 'points': [[[x[0], 1], [x[1], 1]] for x in best_points]}
                
                new_point = (x, y)
                if new_point in best_points:
                    continue
                
                all_ok = True
                for p in best_points:
                    dx = x - p[0]
                    dy = y - p[1]
                    dsq = dx * dx + dy * dy
                    if not is_square(dsq):
                        all_ok = False
                        break
                if not all_ok:
                    continue
                
                candidate = best_points + [new_point]
                if not no_three_collinear(candidate):
                    continue
                if not no_four_concyclic_v2(candidate, D):
                    continue
                
                best_points = candidate
                break
            if len(best_points) > 6:
                break
    
    # Return the best set found
    return {'D': D, 'points': [[[x[0], 1], [x[1], 1]] for x in best_points]}
