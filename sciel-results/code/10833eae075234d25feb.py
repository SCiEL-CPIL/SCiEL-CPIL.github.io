import time
from math import isqrt, gcd
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def time_ok():
        return time.time() - t0 < DEADLINE

    def is_square(n):
        """Check if non-negative integer n is a perfect square."""
        if n < 0:
            return False
        r = isqrt(n)
        return r * r == n

    def is_rational_square(frac):
        """Check if Fraction is a perfect square of a rational."""
        if frac < 0:
            return False
        return is_square(frac.numerator) and is_square(frac.denominator)

    def check_no_three_collinear(pts):
        """pts: list of (x_num, x_den, y_num, y_den). No three collinear."""
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    x1 = Fraction(pts[i][0], pts[i][1])
                    y1 = Fraction(pts[i][2], pts[i][3])
                    x2 = Fraction(pts[j][0], pts[j][1])
                    y2 = Fraction(pts[j][2], pts[j][3])
                    x3 = Fraction(pts[k][0], pts[k][1])
                    y3 = Fraction(pts[k][2], pts[k][3])
                    det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
                    if det == 0:
                        return False
        return True

    def check_no_four_concyclic(pts):
        """pts: list of (x_num, x_den, y_num, y_den). No four concyclic."""
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        x1 = Fraction(pts[i][0], pts[i][1])
                        y1 = Fraction(pts[i][2], pts[i][3])
                        x2 = Fraction(pts[j][0], pts[j][1])
                        y2 = Fraction(pts[j][2], pts[j][3])
                        x3 = Fraction(pts[k][0], pts[k][1])
                        y3 = Fraction(pts[k][2], pts[k][3])
                        x4 = Fraction(pts[l][0], pts[l][1])
                        y4 = Fraction(pts[l][2], pts[l][3])
                        a = x1*x1 + y1*y1
                        b = x2*x2 + y2*y2
                        c = x3*x3 + y3*y3
                        d = x4*x4 + y4*y4
                        # 4x4 determinant using expansion
                        det = (a*(x2*(y3 - y4) - x3*(y2 - y4)) - x1*(b*(y3 - y4) - c*(y2 - y4)) 
                               + y1*(b*(x3 - x4) - c*(x2 - x4)))
                        # Actually let me use the proper formula
                        # |x^2+y^2, x, y, 1|
                        # |x^2+y^2, x, y, 1|
                        # |x^2+y^2, x, y, 1|
                        # |x^2+y^2, x, y, 1|
                        # Determinant = 0 iff concyclic
                        # Use the fact that four points are concyclic iff
                        # the cross ratio is real, or equivalently the determinant vanishes
                        # For integer coords, use:
                        # (x1^2+y1^2)(x2-x4)(y3-y4) - (x1^2+y1^2)(x3-x4)(y2-y4)
                        # - x1[(x2^2+y2^2)(x3-x4) - (x3^2+y3^2)(x2-x4)]
                        # + y1[(x2^2+y2^2)(y3-y4) - (x3^2+y3^2)(y2-y4)]
                        # + (x2^2+y2^2)(x3-x4)(y4-y1) - (x3^2+y3^2)(x2-x4)(y4-y1)
                        # This is getting complex. Let me just compute it directly.
                        
                        # Proper 4x4 det computation
                        # Row i: [xi^2+yi^2, xi, yi, 1]
                        r1 = [x1*x1 + y1*y1, x1, y1, 1]
                        r2 = [x2*x2 + y2*y2, x2, y2, 1]
                        r3 = [x3*x3 + y3*y3, x3, y3, 1]
                        r4 = [x4*x4 + y4*y4, x4, y4, 1]
                        
                        # Expand along first row
                        def det3(m):
                            return (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
                                   - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
                                   + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))
                        
                        M1 = [r2[1:], r3[1:], r4[1:]]
                        M2 = [r2[:1]+r2[2:], r3[:1]+r3[2:], r4[:1]+r4[2:]]
                        M3 = [r2[:2]+r2[3:], r3[:2]+r3[3:], r4[:2]+r4[3:]]
                        M4 = [r2[:3], r3[:3], r4[:3]]
                        
                        det_val = r1[0]*det3(M1) - r1[1]*det3(M2) + r1[2]*det3(M3) - r1[3]*det3(M4)
                        if det_val == 0:
                            return False
        return True

    def validate(pts, D):
        """Full validation."""
        n = len(pts)
        if n < 3:
            return False
        seen = set()
        for p in pts:
            key = (Fraction(p[0], p[1]), Fraction(p[2], p[3]))
            if key in seen:
                return False
            seen.add(key)
        # Check rational distances
        for i in range(n):
            for j in range(i + 1, n):
                dx = Fraction(pts[i][0], pts[i][1]) - Fraction(pts[j][0], pts[j][1])
                dy = Fraction(pts[i][2], pts[i][3]) - Fraction(pts[j][2], pts[j][3])
                d2 = dx * dx + D * dy * dy
                if not is_rational_square(d2):
                    return False
        if not check_no_three_collinear(pts):
            return False
        if not check_no_four_concyclic(pts):
            return False
        return True

    def to_output(pts, D):
        """Convert internal representation to output format."""
        out = []
        for (xn, xd, yn, yd) in pts:
            out.append([[xn, xd], [yn, yd]])
        return {'D': D, 'points': out}

    # Start with seed: 5 points
    seed_pts = [(0, 1, 0, 1), (-56, 1, 33, 1), (-16, 1, 63, 1), (0, 1, 33, 1), (16, 1, 63, 1)]
    D = 1

    best_pts = seed_pts[:]
    best_D = D
    best_count = len(seed_pts)

    if not validate(best_pts, best_D):
        best_pts = [(0,1,0,1), (3,1,0,1), (0,1,4,1)]
        best_count = 3

    # Efficient search: pre-filter candidates by distance to first 2 points
    # For D=1, integer coords, distance squared must be perfect square
    
    def find_addable_points(current_pts, search_range):
        """Find all points in [-search_range, search_range]^2 that have integer distance to all current_pts."""
        n = len(current_pts)
        candidates = []
        for x in range(-search_range, search_range + 1):
            if not time_ok():
                break
            for y in range(-search_range, search_range + 1):
                valid = True
                for i in range(n):
                    dx = x - current_pts[i][0]
                    dy = y - current_pts[i][2]
                    d2 = dx * dx + dy * dy
                    if not is_square(d2):
                        valid = False
                        break
                if valid:
                    candidates.append((x, 1, y, 1))
        return candidates

    # Iteratively add points
    current_pts = best_pts[:]
    current_D = D
    
    # Try to add 6th point with increasing search range
    for sr in [50, 100, 150, 200, 300, 400, 500]:
        if not time_ok():
            break
        if len(current_pts) >= 6:
            break
        candidates = find_addable_points(current_pts, sr)
        for cand in candidates:
            if not time_ok():
                break
            new_pts = current_pts + [cand]
            if check_no_three_collinear(new_pts) and check_no_four_concyclic(new_pts):
                current_pts.append(cand)
                best_pts = current_pts[:]
                best_D = current_D
                best_count = len(current_pts)
                break
        if len(current_pts) > best_count:
            break
    
    # Try to add 7th point
    if time_ok() and len(current_pts) < 7:
        for sr in [100, 200, 300, 400, 500, 600]:
            if not time_ok():
                break
            if len(current_pts) >= 7:
                break
            candidates = find_addable_points(current_pts, sr)
            for cand in candidates:
                if not time_ok():
                    break
                new_pts = current_pts + [cand]
                if check_no_three_collinear(new_pts) and check_no_four_concyclic(new_pts):
                    current_pts.append(cand)
                    best_pts = current_pts[:]
                    best_D = current_D
                    best_count = len(current_pts)
                    break
            if len(current_pts) > best_count:
                break
    
    # Try to add 8th point
    if time_ok() and len(current_pts) < 8:
        for sr in [200, 300, 400, 500, 600, 800]:
            if not time_ok():
                break
            if len(current_pts) >= 8:
                break
            candidates = find_addable_points(current_pts, sr)
            for cand in candidates:
                if not time_ok():
                    break
                new_pts = current_pts + [cand]
                if check_no_three_collinear(new_pts) and check_no_four_concyclic(new_pts):
                    current_pts.append(cand)
                    best_pts = current_pts[:]
                    best_D = current_D
                    best_count = len(current_pts)
                    break
            if len(current_pts) > best_count:
                break

    return to_output(best_pts, best_D)

# EVOLVE-BLOCK-END
