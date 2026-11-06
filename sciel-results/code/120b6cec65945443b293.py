"""
Integral point sets in general position.
Strategy: start from a known 5-point integer configuration, then extend by
parametrising circles through triples of existing points (rational points on a
circle).  All arithmetic is exact (integers).  A fallback local search over
lattice points is used if the algebraic extension stalls.
"""
import time
import random
import math
from itertools import combinations

_DEADLINE = time.time() + 115.0  # safety margin under 120 s

# ---------- exact integer helpers ----------

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _sq_dist(p, q):
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + dy * dy

def _sq_dist_is_square(p, q):
    return _is_square(_sq_dist(p, q))

def _collinear(p, q, r):
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]) == 0

def _det3(a, b, c):
    return (a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]))

def _concyclic(p, q, r, s):
    pts = [p, q, r, s]
    signs = [1, -1, 1, -1]
    total = 0
    for i in range(4):
        rows = [pts[j] for j in range(4) if j != i]
        a = (rows[0][0], rows[0][1], rows[0][0] * rows[0][0] + rows[0][1] * rows[0][1])
        b = (rows[1][0], rows[1][1], rows[1][0] * rows[1][0] + rows[1][1] * rows[1][1])
        c = (rows[2][0], rows[2][1], rows[2][0] * rows[2][0] + rows[2][1] * rows[2][1])
        d = _det3(a, b, c)
        total += signs[i] * d
    return total == 0

def _check_pairwise_rational(pts):
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if not _sq_dist_is_square(pts[i], pts[j]):
                return False
    return True

def _check_no_three_collinear(pts):
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear(pts[i], pts[j], pts[k]):
                    return False
    return True

def _check_no_four_concyclic(pts):
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if _concyclic(pts[i], pts[j], pts[k], pts[l]):
                        return False
    return True

def _is_valid_set(pts):
    if len(pts) < 3:
        return False
    if len(set(tuple(p) for p in pts)) != len(pts):
        return False
    if not _check_pairwise_rational(pts):
        return False
    if not _check_no_three_collinear(pts):
        return False
    if not _check_no_four_concyclic(pts):
        return False
    return True

# ---------- fast incremental add check ----------

def _can_add_point(pts, cand):
    if any(cand == p for p in pts):
        return False
    for p in pts:
        if not _sq_dist_is_square(cand, p):
            return False
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            if _collinear(pts[i], pts[j], cand):
                return False
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _concyclic(pts[i], pts[j], pts[k], cand):
                    return False
    return True

# ---------- circle through three points, rational parametrisation ----------

def _circle_coeffs(p, q, r):
    """
    Circle through p, q, r: x^2 + y^2 + A x + B y + C = 0.
    Returns (A, B, C) as Fractions.
    """
    from fractions import Fraction
    x1, y1 = p
    x2, y2 = q
    x3, y3 = r
    # x^2+y^2 + A x + B y + C = 0
    # A x1 + B y1 + C = -(x1^2+y1^2)
    # A x2 + B y2 + C = -(x2^2+y2^2)
    # A x3 + B y3 + C = -(x3^2+y3^2)
    # Subtract first from second and third:
    # A(x2-x1) + B(y2-y1) = -(x2^2-x1^2) - (y2^2-y1^2)
    # A(x3-x1) + B(y3-y1) = -(x3^2-x1^2) - (y3^2-y1^2)
    a11 = x2 - x1
    a12 = y2 - y1
    b1 = -(x2 * x2 - x1 * x1) - (y2 * y2 - y1 * y1)
    a21 = x3 - x1
    a22 = y3 - y1
    b2 = -(x3 * x3 - x1 * x1) - (y3 * y3 - y1 * y1)
    det = a11 * a22 - a12 * a21
    if det == 0:
        return None
    A = Fraction(b1 * a22 - b2 * a12, det)
    B = Fraction(a11 * b2 - a21 * b1, det)
    C = Fraction(-(x1 * x1 + y1 * y1) - A * x1 - B * y1, 1)
    return (A, B, C)

def _circle_points(p, q, r, t_num, t_den):
    """
    Parametrise circle through p, q, r using rational parameter t = t_num/t_den.
    Returns list of candidate integer points (may be empty).
    Standard rational parametrisation of a circle:
    Given three points, we can use the fact that a circle with rational
    coefficients has rational points parameterised by slope from one point.
    """
    from fractions import Fraction
    coeffs = _circle_coeffs(p, q, r)
    if coeffs is None:
        return []
    A, B, C = coeffs
    # Circle: x^2 + y^2 + A x + B y + C = 0
    # Complete the square: (x + A/2)^2 + (y + B/2)^2 = (A^2+B^2)/4 - C
    # Centre: (-A/2, -B/2), radius^2 = (A^2+B^2)/4 - C
    cx = -A / 2
    cy = -B / 2
    r2 = (A * A + B * B) / 4 - C
    if r2 < 0:
        return []
    # Use p as base point on circle. Parametrise by slope m from p:
    # Line through p with slope m: y - y1 = m(x - x1)
    # Substitute into circle, one root is x1, other gives second intersection.
    # x^2 + y^2 + A x + B y + C = 0
    # y = y1 + m(x - x1)
    # x^2 + (y1 + m(x-x1))^2 + A x + B(y1 + m(x-x1)) + C = 0
    # This is quadratic in x. One root is x1.
    # Let's use the standard trick: if (x1,y1) is on circle, the other
    # intersection with line of slope m from (x1,y1) can be found.
    x1, y1 = p
    m = Fraction(t_num, t_den)
    # Expand: x^2 + (y1 + m(x-x1))^2 + A x + B(y1 + m(x-x1)) + C = 0
    # = x^2 + y1^2 + 2*y1*m*(x-x1) + m^2*(x-x1)^2 + A*x + B*y1 + B*m*(x-x1) + C
    # Let u = x - x1:
    # (x1+u)^2 + (y1 + m*u)^2 + A*(x1+u) + B*(y1+m*u) + C = 0
    # x1^2 + 2*x1*u + u^2 + y1^2 + 2*y1*m*u + m^2*u^2 + A*x1 + A*u + B*y1 + B*m*u + C = 0
    # (x1^2+y1^2+A*x1+B*y1+C) + u*(2*x1 + 2*y1*m + A + B*m) + u^2*(1+m^2) = 0
    # First term is 0 since p is on circle.
    # u*(2*x1 + 2*y1*m + A + B*m) + u^2*(1+m^2) = 0
    # u = 0 (the point p itself) or:
    # u = -(2*x1 + 2*y1*m + A + B*m) / (1 + m^2)
    num_u = -(2 * x1 + 2 * y1 * m + A + B * m)
    den_u = 1 + m * m
    u = num_u / den_u
    x_new = x1 + u
    y_new = y1 + m * u
    # Check if integer
    if x_new.denominator == 1 and y_new.denominator == 1:
        return [(x_new.numerator, y_new.numerator)]
    return []

# ---------- known base ----------

_BASE_POINTS = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]

# ---------- algebraic extension ----------

def _extend_algebraic(pts, time_limit, rng):
    """Try to extend pts by adding points from circle parametrisations."""
    pts = list(pts)
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        n = len(pts)
        # For each triple, try several rational slopes
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if time.time() >= time_limit:
                        break
                    triple = (pts[i], pts[j], pts[k])
                    # Try slopes from a small set of rationals
                    for t_num in range(-12, 13):
                        for t_den in range(1, 13):
                            if t_den == 0:
                                continue
                            if math.gcd(abs(t_num), t_den) != 1:
                                continue
                            cands = _circle_points(triple[0], triple[1], triple[2], t_num, t_den)
                            for cand in cands:
                                if _can_add_point(pts, cand):
                                    pts.append(cand)
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return pts

# ---------- local search fallback ----------

def _extend_local(pts, time_limit, rng):
    """Try to extend pts by random lattice points."""
    pts = list(pts)
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        for _ in range(5000):
            if time.time() >= time_limit:
                break
            x = rng.randint(-200, 200)
            y = rng.randint(-200, 200)
            cand = (x, y)
            if _can_add_point(pts, cand):
                pts.append(cand)
                improved = True
                break
    return pts

def construct():
    start = time.time()
    deadline = start + 110.0
    rng = random.Random(12345)

    pts = list(_BASE_POINTS)

    # Verify base
    if not _is_valid_set(pts):
        pts = [(0, 0), (3, 4), (0, 5)]

    # Phase 1: algebraic extension (circle parametrisation)
    remaining = deadline - time.time()
    if remaining > 5:
        pts = _extend_algebraic(pts, time.time() + remaining * 0.5, rng)

    # Phase 2: local search
    remaining = deadline - time.time()
    if remaining > 3:
        pts = _extend_local(pts, time.time() + remaining * 0.4, rng)

    # Phase 3: more local search with wider range
    remaining = deadline - time.time()
    if remaining > 2:
        pts = _extend_local(pts, time.time() + remaining * 0.3, rng)

    # Final validation
    if not _is_valid_set(pts):
        pts = list(_BASE_POINTS)

    out_points = [[[x, 1], [y, 1]] for x, y in pts]
    return {'D': 1, 'points': out_points}
