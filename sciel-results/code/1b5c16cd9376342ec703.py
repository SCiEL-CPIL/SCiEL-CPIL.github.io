"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction
from math import gcd
from math import isqrt

# EVOLVE-BLOCK-START

def _is_squarefree(n):
    if n < 1:
        return False
    if n == 1:
        return True
    i = 2
    while i * i <= n:
        if n % (i * i) == 0:
            return False
        i += 1
    return True

def _check_no_three_collinear(points):
    """Check that no three points are collinear using exact rational arithmetic."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x_i, y_i = points[i]
                x_j, y_j = points[j]
                x_k, y_k = points[k]
                if (x_j - x_i) * (y_k - y_i) == (x_k - x_i) * (y_j - y_i):
                    return False
    return True

def _check_no_four_concyclic(points, D):
    """Check that no four points are concyclic using exact rational arithmetic."""
    n = len(points)
    # Precompute v_i = x_i^2 + D*y_i^2
    vs = [points[i][0]*points[i][0] + D*points[i][1]*points[i][1] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    a1, b1 = points[i]
                    a2, b2 = points[j]
                    a3, b3 = points[k]
                    a4, b4 = points[l]
                    v1, v2, v3, v4 = vs[i], vs[j], vs[k], vs[l]
                    term1 = v1 * (a2 * (b3 - b4) - a3 * (b2 - b4) + a4 * (b2 - b3))
                    term2 = -a1 * (v2 * (b3 - b4) - v3 * (b2 - b4) + v4 * (b2 - b3))
                    term3 = b1 * (v2 * (a3 - a4) - v3 * (a2 - a4) + v4 * (a2 - a3))
                    term4 = -(v2 * (a3 * b4 - b3 * a4) - v3 * (a2 * b4 - b2 * a4) + v4 * (a2 * b3 - b2 * a3))
                    det = term1 + term2 + term3 + term4
                    if det == 0:
                        return False
    return True

def _is_perfect_square(n):
    """Check if n is a perfect square using integer arithmetic."""
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _check_all_pairwise_distances_rational(points, D):
    """Check that all pairwise distances are rational."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            x_i, y_i = points[i]
            x_j, y_j = points[j]
            dx = x_i - x_j
            dy = y_i - y_j
            dist_sq = dx * dx + D * dy * dy
            if dist_sq <= 0:
                return False
            num = dist_sq.numerator
            den = dist_sq.denominator
            if not _is_perfect_square(num):
                return False
            if not _is_perfect_square(den):
                return False
    return True

def _generate_pythagorean_points(D, max_coord=200):
    """Generate points (a, b) where a^2 + D*b^2 is a perfect square."""
    pts = set()
    pts.add((0, 0))
    for b in range(1, max_coord + 1):
        for a in range(1, max_coord + 1):
            val = a * a + D * b * b
            r = isqrt(val)
            if r * r == val:
                pts.add((a, b))
                pts.add((-a, b))
                pts.add((a, -b))
                pts.add((-a, -b))
    return list(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline, 10s margin
    
    def format_result(points, D):
        result_points = []
        for x, y in points:
            result_points.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
        return {'D': D, 'points': result_points}
    
    best_points = None
    best_D = 1
    best_count = 0
    
    # Try D=1 first with known configs
    configs = [
        [(Fraction(0), Fraction(0)),
         (Fraction(-56), Fraction(33)),
         (Fraction(-16), Fraction(63)),
         (Fraction(0), Fraction(33)),
         (Fraction(16), Fraction(63))],
    ]
    
    for pts in configs:
        if _check_no_three_collinear(pts) and \
           _check_no_four_concyclic(pts, 1) and \
           _check_all_pairwise_distances_rational(pts, 1):
            if len(pts) > best_count:
                best_count = len(pts)
                best_points = pts
                best_D = 1
    
    # Try to extend with D=1 using rational parametrization of circle
    # Points on unit circle: (2t/(1+t^2), (1-t^2)/(1+t^2)) for rational t
    # Scale to get integer coordinates
    if best_points and time.time() < deadline - 20:
        # Generate more candidates using circle parametrization
        circle_pts = set()
        for t_num in range(-30, 31):
            for t_den in range(1, 31):
                if gcd(abs(t_num), t_den) == 1:
                    t = Fraction(t_num, t_den)
                    x = 2 * t / (1 + t * t)
                    y = (1 - t * t) / (1 + t * t)
                    # Scale to clear denominators
                    # x = 2*t_num*t_den / (t_den^2 + t_num^2)
                    # y = (t_den^2 - t_num^2) / (t_den^2 + t_num^2)
                    denom = t_den * t_den + t_num * t_num
                    x_num = 2 * t_num * t_den
                    y_num = t_den * t_den - t_num * t_num
                    # Reduce
                    g1 = gcd(abs(x_num), denom)
                    g2 = gcd(abs(y_num), denom)
                    x_r = Fraction(x_num // g1, denom // g1)
                    y_r = Fraction(y_num // g2, denom // g2)
                    circle_pts.add((x_r, y_r))
        
        # Add integer points
        for a in range(-20, 21):
            for b in range(-20, 21):
                circle_pts.add((Fraction(a), Fraction(b)))
        
        # Try to add points from circle_pts to best_points
        candidates = list(circle_pts)
        # Sort by distance to centroid
        if best_points:
            cx_avg = sum(p[0] for p in best_points) / len(best_points)
            cy_avg = sum(p[1] for p in best_points) / len(best_points)
            candidates.sort(key=lambda c: (c[0] - cx_avg)**2 + (c[1] - cy_avg)**2)
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for candidate in candidates:
                if time.time() > deadline - 10:
                    break
                if any(candidate[0] == p[0] and candidate[1] == p[1] for p in best_points):
                    continue
                # Fast checks
                ok = True
                for i in range(len(best_points)):
                    for j in range(i + 1, len(best_points)):
                        xi, yi = best_points[i]
                        xj, yj = best_points[j]
                        xc, yc = candidate
                        if (xj - xi) * (yc - yi) == (xc - xi) * (yj - yi):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                # Check distances
                for i in range(len(best_points)):
                    xi, yi = best_points[i]
                    xc, yc = candidate
                    dx = xi - xc
                    dy = yi - yc
                    dist_sq = dx * dx + best_D * dy * dy
                    if dist_sq <= 0:
                        ok = False
                        break
                    if not _is_perfect_square(dist_sq.numerator) or not _is_perfect_square(dist_sq.denominator):
                        ok = False
                        break
                if not ok:
                    continue
                # Full check
                new_points = best_points + [candidate]
                if _check_no_three_collinear(new_points) and \
                   _check_no_four_concyclic(new_points, best_D) and \
                   _check_all_pairwise_distances_rational(new_points, best_D):
                    best_points = new_points
                    best_count = len(new_points)
                    improved = True
                    # Remove candidate from list to avoid rechecking
                    candidates = [c for c in candidates if c != candidate]
                    break
    
    # Also try D=2,3,5 with pythagorean points
    for D in [2, 3, 5]:
        if time.time() > deadline - 15:
            break
        pyth_pts = _generate_pythagorean_points(D, 100)
        if len(pyth_pts) < 3:
            continue
        # Try to build a valid set from these points
        # Start with origin, try to add points
        trial = [(Fraction(0), Fraction(0))]
        for pt in pyth_pts:
            if pt == (0, 0):
                continue
            xc, yc = Fraction(pt[0]), Fraction(pt[1])
            # Check collinearity with existing
            ok = True
            for i in range(len(trial)):
                for j in range(i + 1, len(trial)):
                    xi, yi = trial[i]
                    xj, yj = trial[j]
                    if (xj - xi) * (yc - yi) == (xc - xi) * (yj - yi):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            # Check distances
            for i in range(len(trial)):
                xi, yi = trial[i]
                dx = xi - xc
                dy = yi - yc
                dist_sq = dx * dx + D * dy * dy
                if dist_sq <= 0:
                    ok = False
                    break
                if not _is_perfect_square(dist_sq.numerator) or not _is_perfect_square(dist_sq.denominator):
                    ok = False
                    break
            if not ok:
                continue
            new_trial = trial + [(xc, yc)]
            if _check_no_three_collinear(new_trial) and \
               _check_no_four_concyclic(new_trial, D) and \
               _check_all_pairwise_distances_rational(new_trial, D):
                trial = new_trial
            if time.time() > deadline - 15:
                break
        if len(trial) > best_count:
            best_count = len(trial)
            best_points = trial
            best_D = D
    
    if best_points is None:
        # Fallback
        base_points = [(Fraction(0), Fraction(0)),
                       (Fraction(-56), Fraction(33)),
                       (Fraction(-16), Fraction(63)),
                       (Fraction(0), Fraction(33)),
                       (Fraction(16), Fraction(63))]
        return format_result(base_points, 1)
    
    return format_result(best_points, best_D)

# EVOLVE-BLOCK-END
