"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from fractions import Fraction
from itertools import combinations

def _time_ok(deadline, margin=2.0):
    return time.time() < deadline - margin

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _check_collinear(pts):
    """Check if any three points are collinear. Returns True if valid (no three collinear)."""
    for i, j, k in combinations(range(len(pts)), 3):
        x1, y1 = pts[i]
        x2, y2 = pts[j]
        x3, y3 = pts[k]
        # Cross product (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
        cp = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        if cp == 0:
            return False
    return True

def _check_concyclic(pts):
    """Check if any four points are concyclic. Returns True if valid (no four concyclic)."""
    n = len(pts)
    for i, j, k, l in combinations(range(n), 4):
        x1, y1 = pts[i]
        x2, y2 = pts[j]
        x3, y3 = pts[k]
        x4, y4 = pts[l]
        # Circle through i, j, k: determinant condition for concyclicity
        # Using the equation: x^2 + y^2 + Ax + By + C = 0
        # For four points to be concyclic, the determinant of the following matrix must be nonzero:
        # |x1^2+y1^2, x1, y1, 1|
        # |x2^2+y2^2, x2, y2, 1|
        # |x3^2+y3^2, x3, y3, 1|
        # |x4^2+y4^2, x4, y4, 1|
        # If determinant is 0, they are concyclic.
        # Compute determinant using fraction arithmetic for exactness.
        a1, b1, c1 = x1 * x1 + y1 * y1, x1, y1
        a2, b2, c2 = x2 * x2 + y2 * y2, x2, y2
        a3, b3, c3 = x3 * x3 + y3 * y3, x3, y3
        a4, b4, c4 = x4 * x4 + y4 * y4, x4, y4
        
        # Det of 4x4 matrix [[a,b,c,1],[a,b,c,1],[a,b,c,1],[a,b,c,1]]
        # Using cofactor expansion or direct formula.
        # Let's use the standard 4x4 determinant formula.
        # det = a1*(b2*(c3*1 - 1*c4) - c2*(b3*1 - 1*c4) + 1*(b3*c4 - c3*b4))
        #     - b1*(a2*(c3*1 - 1*c4) - c2*(a3*1 - 1*c4) + 1*(a3*c4 - c3*a4))
        #     + c1*(a2*(b3*1 - 1*c4) - b2*(a3*1 - 1*c4) + 1*(a3*c4 - c3*a4))
        #     - 1*(a2*(b3*c4 - c3*b4) - b2*(a3*c4 - c3*a4) + c2*(a3*b4 - b3*a4))
        
        # Actually, let's just compute it directly with a helper.
        def det4(m):
            # m is a 4x4 list of Fractions
            return (m[0][0]*(m[1][1]*m[2][2]*m[3][3] - m[1][1]*m[2][3]*m[3][2] - m[1][2]*m[2][1]*m[3][3] + m[1][2]*m[2][3]*m[3][1] + m[1][3]*m[2][1]*m[3][2] - m[1][3]*m[2][2]*m[3][1])
                  - m[0][1]*(m[1][0]*m[2][2]*m[3][3] - m[1][0]*m[2][3]*m[3][2] - m[1][2]*m[2][0]*m[3][3] + m[1][2]*m[2][3]*m[3][0] + m[1][3]*m[2][0]*m[3][2] - m[1][3]*m[2][2]*m[3][0])
                  + m[0][2]*(m[1][0]*m[2][1]*m[3][3] - m[1][0]*m[2][3]*m[3][1] - m[1][1]*m[2][0]*m[3][3] + m[1][1]*m[2][3]*m[3][0] + m[1][3]*m[2][0]*m[3][1] - m[1][3]*m[2][1]*m[3][0])
                  - m[0][3]*(m[1][0]*m[2][1]*m[3][2] - m[1][0]*m[2][2]*m[3][1] - m[1][1]*m[2][0]*m[3][2] + m[1][1]*m[2][2]*m[3][0] + m[1][2]*m[2][0]*m[3][1] - m[1][2]*m[2][1]*m[3][0]))
        
        m = [[Fraction(a1), Fraction(b1), Fraction(c1), Fraction(1)],
             [Fraction(a2), Fraction(b2), Fraction(c2), Fraction(1)],
             [Fraction(a3), Fraction(b3), Fraction(c3), Fraction(1)],
             [Fraction(a4), Fraction(b4), Fraction(c4), Fraction(1)]]
        
        if det4(m) == 0:
            return False
    return True

def _check_rational_distances(pts, D):
    """Check all pairwise distances are rational. Returns True if valid."""
    for i, j in combinations(range(len(pts)), 2):
        x1, y1 = pts[i]
        x2, y2 = pts[j]
        dx = x1 - x2
        dy = y1 - y2
        # Distance squared = dx^2 + D * dy^2
        d2 = dx * dx + D * dy * dy
        if not _is_square(d2.numerator * d2.denominator):
            # Check if d2 is a perfect square rational
            # d2 = p/q, need p*q to be a perfect square
            pass
        # More precisely: d2 is a rational number. We need sqrt(d2) to be rational.
        # This means d2 = (a/b)^2 for some integers a, b.
        # So d2 * b^2 = a^2, meaning d2 * b^2 is a perfect square integer.
        # Equivalently, in the reduced form d2 = p/q (gcd(p,q)=1), p and q must both be perfect squares.
        p, q = d2.numerator, d2.denominator
        if not (_is_square(p) and _is_square(q)):
            return False
    return True

def _validate(pts, D):
    """Full validation: rational distances, no three collinear, no four concyclic."""
    if len(pts) < 3:
        return False
    # Check distinctness
    if len(set(pts)) != len(pts):
        return False
    if not _check_rational_distances(pts, D):
        return False
    if not _check_collinear(pts):
        return False
    if not _check_concyclic(pts):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s
    
    # Known good 5-point configuration
    best_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_D = 1
    
    # Try to find a 6-point configuration
    # Strategy: Start with the known 5 points and try to add a 6th point
    # Or search for 6 points directly in a bounded region
    
    # First, let's try to extend the 5-point set
    base_pts = best_points
    
    # Search for a 6th point that can be added
    # We'll search in a grid around the existing points
    xs = [p[0] for p in base_pts]
    ys = [p[1] for p in base_pts]
    xmin, xmax = min(xs) - 100, max(xs) + 100
    ymin, ymax = min(ys) - 100, max(ys) + 100
    
    found_6 = False
    for x in range(xmin, xmax + 1):
        if not _time_ok(deadline):
            break
        for y in range(ymin, ymax + 1):
            if not _time_ok(deadline):
                break
            new_pt = (x, y)
            if new_pt in base_pts:
                continue
            candidate = base_pts + [new_pt]
            if _validate(candidate, 1):
                best_points = candidate
                best_D = 1
                found_6 = True
                break
        if found_6:
            break
    
    if not found_6:
        # Try a different approach: search for 6 points from scratch
        # Use a smaller search space
        search_range = 80
        found_6 = False
        for x1 in range(-search_range, search_range + 1):
            if not _time_ok(deadline):
                break
            for y1 in range(-search_range, search_range + 1):
                if not _time_ok(deadline):
                    break
                p1 = (x1, y1)
                for x2 in range(x1 + 1, search_range + 1):
                    if not _time_ok(deadline):
                        break
                    for y2 in range(-search_range, search_range + 1):
                        if not _time_ok(deadline):
                            break
                        p2 = (x2, y2)
                        # Check distance p1-p2 is rational (integer since D=1)
                        dx, dy = x2 - x1, y2 - y1
                        if not _is_square(dx * dx + dy * dy):
                            continue
                        for x3 in range(-search_range, search_range + 1):
                            if not _time_ok(deadline):
                                break
                            for y3 in range(-search_range, search_range + 1):
                                if not _time_ok(deadline):
                                    break
                                p3 = (x3, y3)
                                if p3 == p1 or p3 == p2:
                                    continue
                                # Check distances to p1, p2
                                dx1, dy1 = x3 - x1, y3 - y1
                                dx2, dy2 = x3 - x2, y3 - y2
                                if not _is_square(dx1 * dx1 + dy1 * dy1):
                                    continue
                                if not _is_square(dx2 * dx2 + dy2 * dy2):
                                    continue
                                # Check no three collinear
                                cp = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
                                if cp == 0:
                                    continue
                                for x4 in range(-search_range, search_range + 1):
                                    if not _time_ok(deadline):
                                        break
                                    for y4 in range(-search_range, search_range + 1):
                                        if not _time_ok(deadline):
                                            break
                                        p4 = (x4, y4)
                                        if p4 in (p1, p2, p3):
                                            continue
                                        # Check distances to p1, p2, p3
                                        ok = True
                                        for px, py in (p1, p2, p3):
                                            dx, dy = x4 - px, y4 - py
                                            if not _is_square(dx * dx + dy * dy):
                                                ok = False
                                                break
                                        if not ok:
                                            continue
                                        # Check no three collinear with p4
                                        ok = True
                                        for i, j in combinations(range(3), 2):
                                            pts = [p1, p2, p3]
                                            cp = (pts[j][0] - pts[i][0]) * (y4 - pts[i][1]) - (x4 - pts[i][0]) * (pts[j][1] - pts[i][1])
                                            if cp == 0:
                                                ok = False
                                                break
                                        if not ok:
                                            continue
                                        for x5 in range(-search_range, search_range + 1):
                                            if not _time_ok(deadline):
                                                break
                                            for y5 in range(-search_range, search_range + 1):
                                                if not _time_ok(deadline):
                                                    break
                                                p5 = (x5, y5)
                                                if p5 in (p1, p2, p3, p4):
                                                    continue
                                                ok = True
                                                for px, py in (p1, p2, p3, p4):
                                                    dx, dy = x5 - px, y5 - py
                                                    if not _is_square(dx * dx + dy * dy):
                                                        ok = False
                                                        break
                                                if not ok:
                                                    continue
                                                # Check no three collinear with p5
                                                ok = True
                                                for i, j in combinations(range(4), 2):
                                                    pts = [p1, p2, p3, p4]
                                                    cp = (pts[j][0] - pts[i][0]) * (y5 - pts[i][1]) - (x5 - pts[i][0]) * (pts[j][1] - pts[i][1])
                                                    if cp == 0:
                                                        ok = False
                                                        break
                                                if not ok:
                                                    continue
                                                # Check no four concyclic
                                                candidate = [p1, p2, p3, p4, p5]
                                                if _check_concyclic(candidate):
                                                    best_points = candidate
                                                    best_D = 1
                                                    found_6 = True
                                                    break
                                                if found_6:
                                                    break
                                            if found_6:
                                                break
                                        if found_6:
                                            break
                                    if found_6:
                                        break
                                if found_6:
                                    break
                            if found_6:
                                break
                    if found_6:
                        break
            if found_6:
                break
    
    return {'D': best_D, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
# EVOLVE-BLOCK-END
