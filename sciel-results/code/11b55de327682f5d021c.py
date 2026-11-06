"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import isqrt

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _dist_sq_rational(px, py, qx, qy, D):
    """Check if squared distance between (px,py) and (qx,qy) in (x, y*sqrt(D)) is rational.
    Returns (num, den) of squared distance or None if not rational."""
    dx = px - qx
    dy = py - qy
    # dist^2 = dx^2 + D * dy^2
    val = dx * dx + D * dy * dy
    return val  # Fraction

def _check_dist_rational(points, D):
    """Check all pairwise distances are rational (squared distances are perfect squares of rationals)."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            px, py = points[i]
            qx, qy = points[j]
            dx = px - qx
            dy = py - qy
            d2 = dx * dx + D * dy * dy
            if not _is_square(d2.numerator * d2.denominator):
                return False
    return True

def _no_three_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                # Cross product (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1) == 0
                cp = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                if cp == 0:
                    return False
    return True

def _no_four_concyclic(points, D):
    """Check no four points are concyclic.
    Four points (x1,y1),...,(x4,y4) are concyclic iff the determinant of the circle equation is zero.
    Circle: x^2 + y^2 + Ax + By + C = 0
    For points (xi, yi*sqrt(D)): xi^2 + D*yi^2 + A*xi + B*yi*sqrt(D) + C = 0
    This is tricky. Let's use the algebraic condition.
    
    Actually, for points (xi, yi) where yi is the coefficient of sqrt(D), the point in the plane is (xi, yi*sqrt(D)).
    Four such points are concyclic iff there exists a circle through all four.
    
    A circle in the plane: (x-a)^2 + (y-b)^2 = r^2
    => x^2 + y^2 - 2ax - 2by + a^2 + b^2 - r^2 = 0
    => x^2 + y^2 + A*x + B*y + C = 0
    
    For point (xi, yi*sqrt(D)): xi^2 + D*yi^2 + A*xi + B*yi*sqrt(D) + C = 0
    
    For this to hold for rational A, B, C, we need B*yi*sqrt(D) to be rational for all i, which means either B=0 or all yi=0.
    
    Wait, that's not right. A, B, C can be irrational. Let me reconsider.
    
    Actually, the circle equation with real coefficients: x^2 + y^2 + Ax + By + C = 0.
    Substituting (xi, yi*sqrt(D)): xi^2 + D*yi^2 + A*xi + B*yi*sqrt(D) + C = 0.
    
    For 4 points, we have 4 equations in 3 unknowns (A, B, C). They are consistent iff the 4x4 determinant is zero:
    | xi^2 + D*yi^2  xi  yi*sqrt(D)  1 |
    
    This is hard to check exactly. Let me use a different approach.
    
    Alternative: 4 points are concyclic iff the cross-ratio condition holds, or equivalently, the angle subtended by a chord is equal.
    
    Simpler: use the determinant test. Four points (x1,y1),...,(x4,y4) are concyclic iff:
    det([xi, yi, xi^2+yi^2, 1 for i in 1..4]) = 0
    
    Here yi = yi_coeff * sqrt(D), so yi^2 = D * yi_coeff^2.
    The entries involve sqrt(D). Let me separate rational and irrational parts.
    
    Row i: [xi, yi_coeff*sqrt(D), xi^2 + D*yi_coeff^2, 1]
    
    The determinant will be of the form a + b*sqrt(D) where a, b are rational.
    For it to be zero, we need a=0 and b=0.
    
    This is complex. Let me just use the known 5-point set and try to extend it.
    """
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    # Four points concyclic check
                    # Using the determinant: |xi, yi, xi^2+yi^2, 1| = 0
                    # where yi here is the actual y-coordinate = yi_coeff * sqrt(D)
                    # Let's compute using the algebraic extension
                    x1, y1c = points[i]
                    x2, y2c = points[j]
                    x3, y3c = points[k]
                    x4, y4c = points[l]
                    
                    # y_i = y_ic * sqrt(D)
                    # det = |x1, y1c*sqrt(D), x1^2 + D*y1c^2, 1|
                    #       |x2, y2c*sqrt(D), x2^2 + D*y2c^2, 1|
                    #       |x3, y3c*sqrt(D), x3^2 + D*y3c^2, 1|
                    #       |x4, y4c*sqrt(D), x4^2 + D*y4c^2, 1|
                    
                    # Expand along last column:
                    # = 1 * det3(x1,y1c*sqrt(D),x1^2+D*y1c^2; x2,y2c*sqrt(D),x2^2+D*y2c^2; x3,y3c*sqrt(D),x3^2+D*y3c^2)
                    # - 1 * det3(x1,y1c*sqrt(D),x1^2+D*y1c^2; x2,y2c*sqrt(D),x2^2+D*y2c^2; x4,y4c*sqrt(D),x4^2+D*y4c^2)
                    # + 1 * det3(x1,y1c*sqrt(D),x1^2+D*y1c^2; x3,y3c*sqrt(D),x3^2+D*y3c^2; x4,y4c*sqrt(D),x4^2+D*y4c^2)
                    # - 1 * det3(x2,y2c*sqrt(D),x2^2+D*y2c^2; x3,y3c*sqrt(D),x3^2+D*y3c^2; x4,y4c*sqrt(D),x4^2+D*y4c^2)
                    
                    # Each 3x3 det has form a + b*sqrt(D)
                    # Let me compute each as (a, b) representing a + b*sqrt(D)
                    
                    def det3(p1, p2, p3):
                        """Compute 3x3 det where each point is (x, yc, x^2+D*yc^2) and y=yc*sqrt(D)
                        Returns (a, b) for a + b*sqrt(D)"""
                        x1, y1c, v1 = p1
                        x2, y2c, v2 = p2
                        x3, y3c, v3 = p3
                        # det = x1*(y2c*sqrt(D)*v3 - v2*y3c*sqrt(D)) - y1c*sqrt(D)*(x2*v3 - v2*x3) + v1*(x2*y3c*sqrt(D) - y2c*sqrt(D)*x3)
                        # = x1*sqrt(D)*(y2c*v3 - v2*y3c) - y1c*sqrt(D)*(x2*v3 - v2*x3) + v1*sqrt(D)*(x2*y3c - y2c*x3)
                        # = sqrt(D) * [x1*(y2c*v3 - v2*y3c) - y1c*(x2*v3 - v2*x3) + v1*(x2*y3c - y2c*x3)]
                        # So a=0, b = the bracket
                        b = (x1 * (y2c * v3 - v2 * y3c) - y1c * (x2 * v3 - v2 * x3) + v1 * (x2 * y3c - y2c * x3))
                        return (Fraction(0), b)
                    
                    p1 = (x1, y1c, x1*x1 + D*y1c*y1c)
                    p2 = (x2, y2c, x2*x2 + D*y2c*y2c)
                    p3 = (x3, y3c, x3*x3 + D*y3c*y3c)
                    p4 = (x4, y4c, x4*x4 + D*y4c*y4c)
                    
                    d1 = det3(p1, p2, p3)
                    d2 = det3(p1, p2, p4)
                    d3 = det3(p1, p3, p4)
                    d4 = det3(p2, p3, p4)
                    
                    # Total = d1 - d2 + d3 - d4
                    a = d1[0] - d2[0] + d3[0] - d4[0]
                    b = d1[1] - d2[1] + d3[1] - d4[1]
                    
                    if a == 0 and b == 0:
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Known 5-point set
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    D = 1
    
    # Verify base set
    base_frac = [(Fraction(x), Fraction(y)) for x, y in base_points]
    
    # Try to find a 6th point
    # Search over integer lattice points
    best = base_frac
    
    # Try adding points from a bounded region
    for x in range(-200, 201):
        if time.time() > deadline:
            break
        for y in range(-200, 201):
            if time.time() > deadline:
                break
            new_pt = (Fraction(x), Fraction(y))
            # Check not collinear with any pair
            collinear = False
            for i in range(len(base_frac)):
                for j in range(i+1, len(base_frac)):
                    x1, y1 = base_frac[i]
                    x2, y2 = base_frac[j]
                    cp = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
                    if cp == 0:
                        collinear = True
                        break
                if collinear:
                    break
            if collinear:
                continue
            
            # Check all pairwise distances rational (for D=1, just integer distances)
            all_rational = True
            for pt in base_frac:
                dx = x - pt[0]
                dy = y - pt[1]
                d2 = dx * dx + dy * dy
                if not _is_square(d2.numerator * d2.denominator):
                    all_rational = False
                    break
            if not all_rational:
                continue
            
            # Check no four concyclic
            candidate = base_frac + [new_pt]
            if not _no_four_concyclic(candidate, D):
                continue
            
            # Found a 6-point set!
            best = candidate
            # Try to find even more points
            break
        if len(best) > 5:
            break
    
    # If we found 6 points, try to find a 7th
    if len(best) > 5:
        for x in range(-300, 301):
            if time.time() > deadline:
                break
            for y in range(-300, 301):
                if time.time() > deadline:
                    break
                new_pt = (Fraction(x), Fraction(y))
                collinear = False
                for i in range(len(best)):
                    for j in range(i+1, len(best)):
                        x1, y1 = best[i]
                        x2, y2 = best[j]
                        cp = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
                        if cp == 0:
                            collinear = True
                            break
                    if collinear:
                        break
                if collinear:
                    continue
                all_rational = True
                for pt in best:
                    dx = x - pt[0]
                    dy = y - pt[1]
                    d2 = dx * dx + dy * dy
                    if not _is_square(d2.numerator * d2.denominator):
                        all_rational = False
                        break
                if not all_rational:
                    continue
                candidate = best + [new_pt]
                if not _no_four_concyclic(candidate, D):
                    continue
                best = candidate
                break
            if len(best) > 6:
                break
    
    # Format output
    result_points = []
    for x, y in best:
        result_points.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    
    return {'D': D, 'points': result_points}
# EVOLVE-BLOCK-END
