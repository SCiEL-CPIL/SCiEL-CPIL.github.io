"""Rewritten program: faster, broader search for integral point sets (D=1)."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations
from math import isqrt

_DEADLINE = 138.0  # safety margin before 150s kill


def _is_square(n):
    """Check if non-negative integer n is a perfect square."""
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n


def _rational_dist2(p, q):
    """Squared distance between two points as a Fraction."""
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + dy * dy


def _check_rational_dist(p, q):
    """Check if distance between p and q is rational."""
    d2 = _rational_dist2(p, q)
    num, den = d2.numerator, d2.denominator
    prod = num * den
    return _is_square(prod)


def _collinear(p, q, r):
    """Check if three points are collinear using cross product."""
    dx1 = q[0] - p[0]
    dy1 = q[1] - p[1]
    dx2 = r[0] - p[0]
    dy2 = r[1] - p[1]
    return dx1 * dy2 - dy1 * dx2 == 0


def _concyclic(p, q, r, s):
    """Check if four points are concyclic using determinant of circle equation."""
    def f(pt):
        return (pt[0] * pt[0] + pt[1] * pt[1], pt[0], pt[1], 1)
    A, B, C, D = f(p), f(q), f(r), f(s)
    # 4x4 determinant
    det = (A[0] * (B[1] * (C[2] * D[3] - C[3] * D[2]) - B[2] * (C[1] * D[3] - C[3] * D[1]) + B[3] * (C[1] * D[2] - C[2] * D[1]))
           - A[1] * (B[0] * (C[2] * D[3] - C[3] * D[2]) - B[2] * (C[0] * D[3] - C[3] * D[0]) + B[3] * (C[0] * D[2] - C[2] * D[0]))
           + A[2] * (B[0] * (C[1] * D[3] - C[3] * D[1]) - B[1] * (C[0] * D[3] - C[3] * D[0]) + B[3] * (C[0] * D[1] - C[1] * D[0]))
           - A[3] * (B[0] * (C[1] * D[2] - C[2] * D[1]) - B[1] * (C[0] * D[2] - C[2] * D[0]) + B[2] * (C[0] * D[1] - C[1] * D[0])))
    return det == 0


def _check_valid(points):
    """Full validity check: all pairwise distances rational, no 3 collinear, no 4 concyclic."""
    n = len(points)
    if n < 3:
        return False
    seen = set()
    for i in range(n):
        for j in range(i + 1, n):
            if (points[i][0], points[i][1]) == (points[j][0], points[j][1]):
                return False
            if not _check_rational_dist(points[i], points[j]):
                return False
    for combo in combinations(range(n), 3):
        if _collinear(points[combo[0]], points[combo[1]], points[combo[2]]):
            return False
    for combo in combinations(range(n), 4):
        if _concyclic(points[combo[0]], points[combo[1]], points[combo[2]], points[combo[3]]):
            return False
    return True


def _circle_intersections(p1, p2, r1, r2):
    """Find intersection points of two circles centered at p1, p2 with radii r1, r2.
    Returns list of (Fraction, Fraction) or empty list."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    d2 = dx * dx + dy * dy
    if d2 == 0:
        return []
    # Check if d = sqrt(d2) is rational
    num, den = d2.numerator, d2.denominator
    prod = num * den
    if not _is_square(prod):
        return []
    d_val = Fraction(isqrt(prod), den)

    # Standard circle intersection
    # a = (r1^2 - r2^2 + d^2) / (2d)
    a = (r1 * r1 - r2 * r2 + d2) / (2 * d_val)
    h2 = r1 * r1 - a * a
    if h2 < 0:
        return []
    h_num, h_den = h2.numerator, h2.denominator
    h_prod = h_num * h_den
    if not _is_square(h_prod):
        return []
    h_val = Fraction(isqrt(h_prod), h_den)

    # Midpoint along the line from p1 to p2
    mx = p1[0] + a * dx / d_val
    my = p1[1] + a * dy / d_val

    # Perpendicular offset
    px = -dy / d_val * h_val
    py = dx / d_val * h_val

    pt1 = (mx + px, my + py)
    pt2 = (mx - px, my - py)

    results = []
    for pt in [pt1, pt2]:
        x, y = pt
        if isinstance(x, Fraction) and isinstance(y, Fraction):
            # Check if coordinates are "reasonable" (small enough)
            if abs(x.numerator) <= 10**8 and abs(x.denominator) <= 10**8:
                if abs(y.numerator) <= 10**8 and abs(y.denominator) <= 10**8:
                    results.append(pt)
    return results


def _generate_candidates(current, max_radius=100):
    """Generate candidate points from circle intersections of pairs in current."""
    candidates = set()
    n = len(current)

    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = current[i], current[j]
            d12 = _rational_dist2(p1, p2)
            d12_num, d12_den = d12.numerator, d12.denominator
            d12_prod = d12_num * d12_den
            if not _is_square(d12_prod):
                continue
            d12_val = Fraction(isqrt(d12_prod), d12_den)

            # Try integer distances
            for r1 in range(1, max_radius + 1):
                for r2 in range(1, max_radius + 1):
                    # Triangle inequality
                    if r1 + r2 < d12_val or abs(r1 - r2) > d12_val:
                        continue
                    pts = _circle_intersections(p1, p2, Fraction(r1), Fraction(r2))
                    for pt in pts:
                        candidates.add(pt)

    return list(candidates)


def _filter_candidate(cand, current):
    """Check if cand is valid to add to current set."""
    # Check rational distance to all current points
    for bp in current:
        if not _check_rational_dist(cand, bp):
            return False
    # Check no collinearity with any pair
    for i in range(len(current)):
        for j in range(i + 1, len(current)):
            if _collinear(cand, current[i], current[j]):
                return False
    # Check no concyclicity with any triple
    if len(current) >= 3:
        for combo in combinations(range(len(current)), 3):
            if _concyclic(cand, current[combo[0]], current[combo[1]], current[combo[2]]):
                return False
    return True


def _search_incremental(starting_points, max_points=8, time_limit=130.0):
    """Incrementally add points using circle intersections."""
    start_time = time.time()
    best = list(starting_points)
    current = list(starting_points)

    for target_n in range(len(current) + 1, max_points + 1):
        if time.time() - start_time > time_limit:
            break

        # Generate candidates
        candidates = _generate_candidates(current, max_radius=80)

        # Filter and try to add
        found = False
        for cand in candidates:
            if time.time() - start_time > time_limit:
                break
            if cand in current:
                continue
            if _filter_candidate(cand, current):
                new_set = current + [cand]
                if _check_valid(new_set):
                    current = new_set
                    found = True
                    if len(current) > len(best):
                        best = list(current)
                    break

        if not found:
            # Try with larger radius
            if time.time() - start_time < time_limit - 10:
                candidates2 = _generate_candidates(current, max_radius=150)
                for cand in candidates2:
                    if time.time() - start_time > time_limit:
                        break
                    if cand in current:
                        continue
                    if _filter_candidate(cand, current):
                        new_set = current + [cand]
                        if _check_valid(new_set):
                            current = new_set
                            found = True
                            if len(current) > len(best):
                                best = list(current)
                            break
            if not found:
                break

    return best


def construct():
    start_time = time.time()

    # Known valid 5-point set (integral point set)
    base_points = [(Fraction(0), Fraction(0)), (Fraction(-56), Fraction(33)),
                   (Fraction(-16), Fraction(63)), (Fraction(0), Fraction(33)),
                   (Fraction(16), Fraction(63))]

    best = base_points

    # Try to extend from the base set
    try:
        extended = _search_incremental(base_points, max_points=8, time_limit=125.0)
        if len(extended) > len(best):
            best = extended
    except Exception:
        pass

    # If we still only have 5, try a different starting configuration
    if len(best) <= 5 and time.time() - start_time < 100:
        # Try another known small set
        alt_base = [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(4)),
                    (Fraction(0), Fraction(4)), (Fraction(3), Fraction(8)),
                    (Fraction(6), Fraction(4))]
        if _check_valid(alt_base):
            try:
                extended_alt = _search_incremental(alt_base, max_points=8, time_limit=120.0)
                if len(extended_alt) > len(best):
                    best = extended_alt
            except Exception:
                pass

    # Try yet another configuration if still stuck
    if len(best) <= 5 and time.time() - start_time < 80:
        # A different 5-point set
        alt_base2 = [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)),
                     (Fraction(0), Fraction(1)), (Fraction(1), Fraction(1)),
                     (Fraction(1), Fraction(2))]
        if _check_valid(alt_base2):
            try:
                extended_alt2 = _search_incremental(alt_base2, max_points=8, time_limit=110.0)
                if len(extended_alt2) > len(best):
                    best = extended_alt2
            except Exception:
                pass

    # Format output
    points_out = []
    for pt in best:
        x, y = pt
        points_out.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])

    return {'D': 1, 'points': points_out}
# EVOLVE-BLOCK-END
