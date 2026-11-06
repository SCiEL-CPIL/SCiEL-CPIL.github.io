"""Integral point sets in general position via circle parametrization + Möbius transform."""
from fractions import Fraction
from math import isqrt, gcd
from itertools import combinations

def _circle_point(t):
    """Rational parametrization of unit circle: ((1-t^2)/(1+t^2), 2t/(1+t^2))."""
    t2 = t * t
    denom = 1 + t2
    x = (1 - t2) / denom
    y = 2 * t / denom
    return (x, y)

def _mobius_transform_complex(z, a, b, c, d):
    """Apply Möbius transform z -> (a*z + b)/(c*z + d) where z = x + iy."""
    x, y = z
    # c*z + d = c*(x+iy) + d = (cx+d) + i*cy
    denom_re = c * x + d
    denom_im = c * y
    denom_sq = denom_re * denom_re + denom_im * denom_im
    if denom_sq == 0:
        return None
    # (a*z + b) = (a*(x+iy) + b) = (ax+b) + i*ay
    num_re = a * x + b
    num_im = a * y
    # (num_re + i*num_im) / (denom_re + i*denom_im) = 
    # ((num_re*denom_re + num_im*denom_im) + i*(num_im*denom_re - num_re*denom_im)) / denom_sq
    new_x = (num_re * denom_re + num_im * denom_im) / denom_sq
    new_y = (num_im * denom_re - num_re * denom_im) / denom_sq
    return (new_x, new_y)

def _is_rational_square(d2):
    """Check if a Fraction d2 is a perfect square of a rational."""
    num = d2.numerator
    den = d2.denominator
    if num < 0 or den < 0:
        return False
    g = gcd(num, den)
    num_s = num // g
    den_s = den // g
    sqrt_num = isqrt(num_s)
    sqrt_den = isqrt(den_s)
    return sqrt_num * sqrt_num == num_s and sqrt_den * sqrt_den == den_s

def _check_general_position(points):
    """Check no three collinear and no four concyclic using exact arithmetic."""
    n = len(points)
    
    # Check no three collinear
    for i, j, k in combinations(range(n), 3):
        x1, y1 = points[i]
        x2, y2 = points[j]
        x3, y3 = points[k]
        cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        if cross == 0:
            return False
    
    # Check no four concyclic
    for i, j, k, l in combinations(range(n), 4):
        x1, y1 = points[i]
        x2, y2 = points[j]
        x3, y3 = points[k]
        x4, y4 = points[l]
        
        # Four points concyclic iff determinant is zero
        # |x1^2+y1^2, x1, y1, 1|
        # |x2^2+y2^2, x2, y2, 1|
        # |x3^2+y3^2, x3, y3, 1| = 0
        # |x4^2+y4^2, x4, y4, 1|
        
        r1 = x1*x1 + y1*y1
        r2 = x2*x2 + y2*y2
        r3 = x3*x3 + y3*y3
        r4 = x4*x4 + y4*y4
        
        # 4x4 determinant by expansion
        # |r1 x1 y1 1|
        # |r2 x2 y2 1|
        # |r3 x3 y3 1|
        # |r4 x4 y4 1|
        
        # Expand along last column
        # = 1*det([[r2,x2,y2],[r3,x3,y3],[r4,x4,y4]])
        #   - 1*det([[r1,x1,y1],[r3,x3,y3],[r4,x4,y4]])
        #   + 1*det([[r1,x1,y1],[r2,x2,y2],[r4,x4,y4]])
        #   - 1*det([[r1,x1,y1],[r2,x2,y2],[r3,x3,y3]])
        
        def det3(m):
            return (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
                   - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
                   + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))
        
        det1 = det3([[r2,x2,y2],[r3,x3,y3],[r4,x4,y4]])
        det2 = det3([[r1,x1,y1],[r3,x3,y3],[r4,x4,y4]])
        det3v = det3([[r1,x1,y1],[r2,x2,y2],[r4,x4,y4]])
        det4v = det3([[r1,x1,y1],[r2,x2,y2],[r3,x3,y3]])
        
        if det1 - det2 + det3v - det4v == 0:
            return False
    
    return True

def _all_distances_rational(points):
    """Check all pairwise squared distances are perfect squares of rationals."""
    for i, j in combinations(range(len(points)), 2):
        dx = points[i][0] - points[j][0]
        dy = points[i][1] - points[j][1]
        d2 = dx * dx + dy * dy
        if not _is_rational_square(d2):
            return False
    return True

def _find_best_set():
    """Search for a large set of points with rational distances in general position."""
    best_points = None
    best_size = 0
    
    # Generate many candidate t-values for circle parametrization
    t_candidates = []
    # Integer t values
    for t in range(-20, 21):
        t_candidates.append(Fraction(t, 1))
    # Fractional t values with small denominators
    for den in range(2, 10):
        for num in range(-den*5, den*5+1):
            if gcd(abs(num), den) == 1:
                t_candidates.append(Fraction(num, den))
    
    # Remove duplicates and sort
    t_candidates = sorted(set(t_candidates), key=lambda t: float(t))
    
    # Try various Möbius transformations
    # Key insight: Möbius transforms with rational coefficients map the unit circle
    # to another circle/line, and preserve the property that pairwise distances
    # between image points are rational IF the original points have rational distances.
    # But we need to verify this for each case.
    
    mobius_params = []
    # Simple transformations: z -> z (identity), z -> -z, z -> 1/z (inversion)
    # z -> (z+1)/(z-1), etc.
    for a in [-2, -1, 0, 1, 2]:
        for b in [-2, -1, 0, 1, 2]:
            for c in [-2, -1, 0, 1, 2]:
                for d in [-2, -1, 0, 1, 2]:
                    # Skip if determinant is zero
                    if a*d - b*c == 0:
                        continue
                    # Skip if c=0 and d=0 (degenerate)
                    if c == 0 and d == 0:
                        continue
                    mobius_params.append((a, b, c, d))
    
    for (a, b, c, d) in mobius_params:
        # Transform all candidate points
        transformed_points = []
        for t in t_candidates:
            p = _circle_point(t)
            tp = _mobius_transform_complex(p, a, b, c, d)
            if tp is not None:
                transformed_points.append(tp)
        
        if len(transformed_points) < 3:
            continue
        
        # Check distinctness
        seen = set()
        distinct_points = []
        for p in transformed_points:
            key = (p[0].numerator, p[0].denominator, p[1].numerator, p[1].denominator)
            if key not in seen:
                seen.add(key)
                distinct_points.append(p)
        
        if len(distinct_points) < 3:
            continue
        
        # Try greedy selection: add points one by one if they maintain general position
        # and rational distances
        selected = []
        for p in distinct_points:
            candidate = selected + [p]
            if len(candidate) < 3:
                selected = candidate
                continue
            # Check if adding this point maintains all constraints
            # First check rational distances from new point to all existing
            ok = True
            for q in selected:
                dx = p[0] - q[0]
                dy = p[1] - q[1]
                d2 = dx*dx + dy*dy
                if not _is_rational_square(d2):
                    ok = False
                    break
            if not ok:
                continue
            # Check no three collinear with new point
            if len(selected) >= 2:
                for i in range(len(selected)):
                    for j in range(i+1, len(selected)):
                        x1, y1 = selected[i]
                        x2, y2 = selected[j]
                        cross = (x2-x1)*(p[1]-y1) - (y2-y1)*(p[0]-x1)
                        if cross == 0:
                            ok = False
                            break
                    if not ok:
                        break
            if not ok:
                continue
            # Check no four concyclic with new point
            if len(selected) >= 3:
                for i in range(len(selected)):
                    for j in range(i+1, len(selected)):
                        for k in range(j+1, len(selected)):
                            x1, y1 = selected[i]
                            x2, y2 = selected[j]
                            x3, y3 = selected[k]
                            x4, y4 = p
                            r1 = x1*x1 + y1*y1
                            r2 = x2*x2 + y2*y2
                            r3 = x3*x3 + y3*y3
                            r4 = x4*x4 + y4*y4
                            def det3(m):
                                return (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
                                       - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
                                       + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))
                            det1 = det3([[r2,x2,y2],[r3,x3,y3],[r4,x4,y4]])
                            det2 = det3([[r1,x1,y1],[r3,x3,y3],[r4,x4,y4]])
                            det3v = det3([[r1,x1,y1],[r2,x2,y2],[r4,x4,y4]])
                            det4v = det3([[r1,x1,y1],[r2,x2,y2],[r3,x3,y3]])
                            if det1 - det2 + det3v - det4v == 0:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
            if not ok:
                continue
            selected = candidate
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_points = selected
    
    return best_points

def construct():
    """Find a set of integral points in general position with rational distances."""
    # First try the search
    points = _find_best_set()
    
    if points is None or len(points) < 3:
        # Fallback to the known 5-point set
        points = [(Fraction(0, 1), Fraction(0, 1)),
                  (Fraction(-56, 1), Fraction(33, 1)),
                  (Fraction(-16, 1), Fraction(63, 1)),
                  (Fraction(0, 1), Fraction(33, 1)),
                  (Fraction(16, 1), Fraction(63, 1))]
    
    # Convert to required format
    result_points = []
    for x, y in points:
        result_points.append([
            [x.numerator, x.denominator],
            [y.numerator, y.denominator]
        ])
    
    return {'D': 1, 'points': result_points}
