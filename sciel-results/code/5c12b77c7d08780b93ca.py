import time
import math
from itertools import combinations, product
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # 2s safety margin before 150s kill

    def is_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    def dist_sq(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx*dx + dy*dy

    def is_collinear(p1, p2, p3):
        return (p2[0]-p1[0])*(p3[1]-p1[1]) != (p3[0]-p1[0])*(p2[1]-p1[1])

    def is_concyclic(p1, p2, p3, p4):
        x1,y1 = p1; x2,y2 = p2; x3,y3 = p3; x4,y4 = p4
        a1 = x1*x1+y1*y1; a2 = x2*x2+y2*y2
        a3 = x3*x3+y3*y3; a4 = x4*x4+y4*y4
        # Determinant of the circle equation
        # x^2+y^2+Dx+Ey+F = 0
        # Using 4 points, the determinant must be non-zero for non-concyclic
        # det of:
        # [x_i, y_i, x_i^2+y_i^2, 1] for i=1..4
        det = (x2-x1)*((y3-y1)*(a4-a1) - (a3-a1)*(y4-y1)) \
            - (y2-y1)*((x3-x1)*(a4-a1) - (a3-a1)*(x4-x1)) \
            + (a2-a1)*((x3-x1)*(y4-y1) - (y3-y1)*(x4-x1))
        return det != 0

    def validate_set(points):
        n = len(points)
        # Check all pairwise distances are integer
        for i in range(n):
            for j in range(i+1, n):
                d = dist_sq(points[i], points[j])
                if not is_square(d):
                    return False
        # Check no three collinear
        for i, j, k in combinations(range(n), 3):
            if not is_collinear(points[i], points[j], points[k]):
                return False
        # Check no four concyclic
        for i, j, k, l in combinations(range(n), 4):
            if not is_concyclic(points[i], points[j], points[k], points[l]):
                return False
        return True

    def place_point(x, y1, y2, d1, d2, sign):
        """Place point P=(x,y) given distance d1 to (0,y1) and d2 to (0,y2)."""
        # d1^2 = x^2 + (y-y1)^2
        # d2^2 = x^2 + (y-y2)^2
        # Subtract: d1^2 - d2^2 = (y-y1)^2 - (y-y2)^2 = -2y(y1-y2) + (y1^2-y2^2)
        num = d1*d1 - d2*d2 + y1*y1 - y2*y2
        den = 2*(y1-y2)
        if den == 0:
            return None
        y_val = Fraction(num, den)
        # x^2 = d1^2 - (y_val-y1)^2
        x2 = d1*d1 - (y_val - y1)**2
        if x2 < 0:
            return None
        x_num, x_den = x2.numerator, x2.denominator
        if x_den != 1:
            return None
        if x_num < 0:
            return None
        if not is_square(x_num):
            return None
        x_abs = math.isqrt(x_num)
        if sign > 0:
            return (Fraction(x_abs, 1), y_val)
        else:
            return (Fraction(-x_abs, 1), y_val)

    def try_add_point(points, x, d1, d2, sign):
        """Try to add a new point at x-coordinate x, with distances d1 to points[0] and d2 to points[1]."""
        p0 = points[0]
        p1 = points[1]
        # We want new point P = (x, y)
        # d1^2 = (x - p0[0])^2 + (y - p0[1])^2
        # d2^2 = (x - p1[0])^2 + (y - p1[1])^2
        # This is more complex. Let me use a different approach.
        pass

    # Strategy: Build from known Pythagorean distance sets
    # Start with small valid sets and try to extend by searching for compatible distances
    
    # A key insight: if we have points with integer coordinates, we need integer distances.
    # The hardest part is finding a set of integer distances that are geometrically realizable.
    
    # Let me try a different approach: use the fact that we can scale.
    # Actually, let me just try to find a 6-point set by brute force but smarter.
    
    # Known: the 5-point seed works. Let me try to extend it.
    seed = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Verify seed
    if not validate_set(seed):
        # Find a minimal valid set
        seed = [(0,0), (3,4)]
        # Extend greedily
        current = seed
        for attempt in range(50):
            if time.time() > deadline - 20:
                break
            found = False
            # Try to add a point
            for x in range(-200, 201):
                if time.time() > deadline - 20:
                    break
                for y in range(-200, 201):
                    p = (x, y)
                    if p in current:
                        continue
                    # Check distances
                    ok = True
                    for q in current:
                        d = dist_sq(p, q)
                        if not is_square(d):
                            ok = False
                            break
                    if not ok:
                        continue
                    # Check collinearity
                    for i, j in combinations(range(len(current)), 2):
                        if not is_collinear(p, current[i], current[j]):
                            ok = False
                            break
                    if not ok:
                        continue
                    # Check concyclicity
                    if len(current) >= 3:
                        for i, j, k in combinations(range(len(current)), 3):
                            if not is_concyclic(p, current[i], current[j], current[k]):
                                ok = False
                                break
                    if ok:
                        current = current + [p]
                        found = True
                        break
                if found:
                    break
            if not found:
                break
        seed = current

    best_set = list(seed)
    
    # Now try to extend best_set using distance-based search
    # For each candidate new point, we need distances to all existing points to be integers
    # Let's search by trying all integer coordinate points in a bounded region
    
    # Precompute for speed
    def can_add_point(p, existing):
        # p is (x, y)
        # Check distances
        for q in existing:
            d = dist_sq(p, q)
            if not is_square(d):
                return False
        # Check collinearity
        for i, j in combinations(range(len(existing)), 2):
            if not is_collinear(p, existing[i], existing[j]):
                return False
        # Check concyclicity
        if len(existing) >= 3:
            for i, j, k in combinations(range(len(existing)), 3):
                if not is_concyclic(p, existing[i], existing[j], existing[k]):
                    return False
        return True

    # Try to extend
    for round_num in range(10):
        if time.time() > deadline - 10:
            break
        added = False
        # Search in an expanding box
        limit = 100 + round_num * 50
        for x in range(-limit, limit + 1):
            if time.time() > deadline - 10:
                break
            for y in range(-limit, limit + 1):
                if time.time() > deadline - 10:
                    break
                p = (x, y)
                if p in best_set:
                    continue
                if can_add_point(p, best_set):
                    best_set = best_set + [p]
                    added = True
                    break
            if added:
                break
        if not added:
            # Try a different region or break
            if round_num > 2:
                break

    # Final verification
    if not validate_set(best_set):
        # Fallback to a minimal valid set
        best_set = [(0, 0), (3, 4)]

    # Format output
    D = 1
    points_formatted = []
    for (x, y) in best_set:
        # x and y are integers (or Fractions that are integers)
        x_num = x.numerator if isinstance(x, Fraction) else x
        x_den = x.denominator if isinstance(x, Fraction) else 1
        y_num = y.numerator if isinstance(y, Fraction) else y
        y_den = y.denominator if isinstance(y, Fraction) else 1
        points_formatted.append([[x_num, x_den], [y_num, y_den]])

    return {'D': D, 'points': points_formatted}
