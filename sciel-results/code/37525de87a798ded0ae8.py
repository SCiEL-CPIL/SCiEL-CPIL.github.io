import time
import math
from itertools import combinations, product
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    def is_square(n):
        if n < 0:
            return False
        if isinstance(n, Fraction):
            n = n.numerator * n.denominator
            if n < 0:
                return False
        r = math.isqrt(n)
        return r * r == n

    def dist_sq(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx*dx + dy*dy

    def check_no_three_collinear(points):
        n = len(points)
        for i, j, k in combinations(range(n), 3):
            p1, p2, p3 = points[i], points[j], points[k]
            # Collinear if cross product is zero
            cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1])
            if cross == 0:
                return False
        return True

    def check_no_four_concyclic(points):
        n = len(points)
        for i, j, k, l in combinations(range(n), 4):
            p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
            x1,y1 = p1; x2,y2 = p2; x3,y3 = p3; x4,y4 = p4
            a1 = x1*x1+y1*y1; a2 = x2*x2+y2*y2
            a3 = x3*x3+y3*y3; a4 = x4*x4+y4*y4
            # Determinant of circle equation matrix
            # det of [x, y, x^2+y^2, 1] for 4 points
            # Non-concyclic means det != 0
            det = (x2-x1)*((y3-y1)*(a4-a1) - (a3-a1)*(y4-y1)) \
                - (y2-y1)*((x3-x1)*(a4-a1) - (a3-a1)*(x4-x1)) \
                + (a2-a1)*((x3-x1)*(y4-y1) - (y3-y1)*(x4-x1))
            if det == 0:
                return False
        return True

    def validate_set(points):
        n = len(points)
        if n < 2:
            return False
        # Check all pairwise distances squared are perfect squares (rational distances)
        for i in range(n):
            for j in range(i+1, n):
                d = dist_sq(points[i], points[j])
                if not is_square(d):
                    return False
        # Check no three collinear
        if not check_no_three_collinear(points):
            return False
        # Check no four concyclic
        if not check_no_four_concyclic(points):
            return False
        return True

    def can_add_point(p, existing):
        """Check if point p can be added to existing set."""
        # Check distances
        for q in existing:
            d = dist_sq(p, q)
            if not is_square(d):
                return False
        # Check collinearity with any pair
        for i, j in combinations(range(len(existing)), 2):
            cross = (existing[i][0]-p[0])*(existing[j][1]-p[1]) - (existing[j][0]-p[0])*(existing[i][1]-p[1])
            if cross == 0:
                return False
        # Check concyclicity with any triple
        if len(existing) >= 3:
            for i, j, k in combinations(range(len(existing)), 3):
                p1, p2, p3, p4 = existing[i], existing[j], existing[k], p
                x1,y1 = p1; x2,y2 = p2; x3,y3 = p3; x4,y4 = p4
                a1 = x1*x1+y1*y1; a2 = x2*x2+y2*y2
                a3 = x3*x3+y3*y3; a4 = x4*x4+y4*y4
                det = (x2-x1)*((y3-y1)*(a4-a1) - (a3-a1)*(y4-y1)) \
                    - (y2-y1)*((x3-x1)*(a4-a1) - (a3-a1)*(x4-x1)) \
                    + (a2-a1)*((x3-x1)*(y4-y1) - (y3-y1)*(x4-x1))
                if det == 0:
                    return False
        return True

    # Start with a known valid 5-point set
    # Points: (0,0), (-56,33), (-16,63), (0,33), (16,63)
    # All distances are integers, no three collinear, no four concyclic
    seed = [(Fraction(0), Fraction(0)),
            (Fraction(-56), Fraction(33)),
            (Fraction(-16), Fraction(63)),
            (Fraction(0), Fraction(33)),
            (Fraction(16), Fraction(63))]

    if not validate_set(seed):
        # Fallback minimal set
        seed = [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(4))]

    best_set = list(seed)

    # Strategy 1: Brute force search for additional points in expanding regions
    # Use integer coordinates for simplicity (D=1)
    def try_add_integer_point(best_set, max_coord):
        """Try to add an integer point to the set."""
        for x in range(-max_coord, max_coord + 1):
            if time.time() > deadline - 5:
                return None
            for y in range(-max_coord, max_coord + 1):
                if time.time() > deadline - 5:
                    return None
                p = (Fraction(x), Fraction(y))
                # Skip if already in set
                if any(p[0] == q[0] and p[1] == q[1] for q in best_set):
                    continue
                if can_add_point(p, best_set):
                    return p
        return None

    # Strategy 2: Parameterized circle approach
    # Given 3 points, find a circle through them, then parameterize rational points on it
    def circle_through_3(p1, p2, p3):
        """Find circle through 3 points. Returns (cx, cy, r^2) or None."""
        x1,y1 = p1; x2,y2 = p2; x3,y3 = p3
        # Circle: x^2+y^2+Dx+Ey+F=0
        # x1^2+y1^2+D*x1+E*y1+F=0
        # x2^2+y2^2+D*x2+E*y2+F=0
        # x3^2+y3^2+D*x3+E*y3+F=0
        a1 = x1*x1+y1*y1; a2 = x2*x2+y2*y2; a3 = x3*x3+y3*y3
        # Subtract first from second and third
        # D*(x2-x1) + E*(y2-y1) = a1-a2
        # D*(x3-x1) + E*(y3-y1) = a1-a3
        det = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
        if det == 0:
            return None  # Collinear
        D_val = ((a1-a2)*(y3-y1) - (a1-a3)*(y2-y1)) / det
        E_val = ((x2-x1)*(a1-a3) - (x3-x1)*(a1-a2)) / det
        F_val = -a1 - D_val*x1 - E_val*y1
        cx = -D_val / 2
        cy = -E_val / 2
        r2 = cx*cx + cy*cy - F_val
        return (cx, cy, r2)

    def rational_points_on_circle(cx, cy, r2, known_points, max_t_num=200, max_t_den=200):
        """Find rational points on circle using parameterization."""
        # First, find one rational point on the circle (use a known point)
        # Circle: (x-cx)^2 + (y-cy)^2 = r2
        # Parameterize using a line through a known rational point
        # If P0 is on circle, line through P0 with slope t: y-y0 = t(x-x0)
        # Substitute into circle equation to get quadratic in x
        # One root is x0, other root gives new point
        
        # Use the first known point as base
        x0, y0 = known_points[0]
        
        found = []
        for t_num in range(-max_t_num, max_t_num + 1):
            if time.time() > deadline - 5:
                break
            for t_den in range(1, max_t_den + 1):
                if time.time() > deadline - 5:
                    break
                t = Fraction(t_num, t_den)
                # Line: y - y0 = t*(x - x0) => y = t*x + (y0 - t*x0)
                b = y0 - t*x0
                # (x-cx)^2 + (t*x+b-cy)^2 = r2
                # x^2 - 2*cx*x + cx^2 + t^2*x^2 + 2*t*b*x + b^2 - 2*cy*t*x - 2*cy*b + cy^2 = r2
                # (1+t^2)*x^2 + (-2*cx + 2*t*b - 2*cy*t)*x + (cx^2 + b^2 - 2*cy*b + cy^2 - r2) = 0
                A = 1 + t*t
                B = -2*cx + 2*t*b - 2*cy*t
                C = cx*cx + b*b - 2*cy*b + cy*cy - r2
                
                # We know x0 is one root, so other root x1 = C/(A*x0) ... no
                # Product of roots = C/A, sum = -B/A
                # x0 + x1 = -B/A => x1 = -B/A - x0
                x1 = -B/A - x0
                y1 = t*x1 + b
                
                # Check if this is a valid new point
                p_new = (x1, y1)
                # Check it's not the same as any known point
                is_new = True
                for kp in known_points:
                    if p_new[0] == kp[0] and p_new[1] == kp[1]:
                        is_new = False
                        break
                if not is_new:
                    continue
                
                # Check rationality (should be automatic from rational arithmetic)
                # Check if distances to all existing points are rational
                valid = True
                for q in known_points:
                    d = dist_sq(p_new, q)
                    if not is_square(d):
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check collinearity
                for i, j in combinations(range(len(known_points)), 2):
                    cross = (known_points[i][0]-p_new[0])*(known_points[j][1]-p_new[1]) - (known_points[j][0]-p_new[0])*(known_points[i][1]-p_new[1])
                    if cross == 0:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check concyclicity with any triple
                if len(known_points) >= 3:
                    for i, j, k in combinations(range(len(known_points)), 3):
                        p1, p2, p3, p4 = known_points[i], known_points[j], known_points[k], p_new
                        xx1,yy1 = p1; xx2,yy2 = p2; xx3,yy3 = p3; xx4,yy4 = p4
                        aa1 = xx1*xx1+yy1*yy1; aa2 = xx2*xx2+yy2*yy2
                        aa3 = xx3*xx3+yy3*yy3; aa4 = xx4*xx4+yy4*yy4
                        det = (xx2-xx1)*((yy3-yy1)*(aa4-aa1) - (aa3-aa1)*(yy4-yy1)) \
                            - (yy2-yy1)*((xx3-xx1)*(aa4-aa1) - (aa3-aa1)*(xx4-xx1)) \
                            + (aa2-aa1)*((xx3-xx1)*(yy4-yy1) - (yy3-yy1)*(xx4-xx1))
                        if det == 0:
                            valid = False
                            break
                    if not valid:
                        continue
                
                found.append(p_new)
        return found

    # Try to extend the set
    # First try integer brute force
    for max_coord in [100, 200, 400, 600]:
        if time.time() > deadline - 10:
            break
        p = try_add_integer_point(best_set, max_coord)
        if p is not None:
            best_set.append(p)
    
    # Now try circle parameterization
    # For each triple of existing points, try to find new points on their circle
    triples = list(combinations(range(len(best_set)), 3))
    for idx, (i, j, k) in enumerate(triples):
        if time.time() > deadline - 5:
            break
        circle = circle_through_3(best_set[i], best_set[j], best_set[k])
        if circle is None:
            continue
        cx, cy, r2 = circle
        new_points = rational_points_on_circle(cx, cy, r2, best_set, max_t_num=100, max_t_den=100)
        for np in new_points:
            if time.time() > deadline - 5:
                break
            if can_add_point(np, best_set):
                best_set.append(np)
                break  # Add one at a time, re-evaluate

    # Final verification
    if not validate_set(best_set):
        # Fallback to seed
        best_set = list(seed)

    # Format output
    D = 1
    points_formatted = []
    for (x, y) in best_set:
        x_num = x.numerator if isinstance(x, Fraction) else x
        x_den = x.denominator if isinstance(x, Fraction) else 1
        y_num = y.numerator if isinstance(y, Fraction) else y
        y_den = y.denominator if isinstance(y, Fraction) else 1
        points_formatted.append([[x_num, x_den], [y_num, y_den]])

    return {'D': D, 'points': points_formatted}
