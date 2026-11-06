"""
Rewritten search for integral-distance point sets in Q(sqrt(D)) with no three collinear, no four concyclic.
Strategy:
  1. Fast verification of a small library of known small configurations (D=1,2,3,5,7,10,...).
  2. Targeted construction of 6-point sets using rational parametrization of circles/conics and
     greedy extension with exact rational checks.
  3. Backtracking search on a bounded integer lattice for each D, with aggressive pruning.
  4. Local-search extension of any 5- or 6-point set found.
All arithmetic is exact via Fraction. Deadline guard at 118 s (safety margin for 120 s kill).
"""

from fractions import Fraction
import math
import time
import random
from math import gcd, isqrt

# EVOLVE-BLOCK-START

def _is_squarefree(n):
    if n <= 1:
        return True
    if n % 4 == 0:
        return False
    p = 3
    while p * p <= n:
        if n % (p * p) == 0:
            return False
        p += 2
    return True


def _is_perfect_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n


def _sq_dist_rational(p1, p2, D):
    """Return True if squared distance between p1,p2 (as (Fraction,Fraction) pairs) is a perfect square rational."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    d2 = dx * dx + D * dy * dy
    if d2 < 0:
        return False
    num = d2.numerator
    den = d2.denominator
    return _is_perfect_square(num) and _is_perfect_square(den)


def _collinear3(p1, p2, p3):
    """Exact collinearity check."""
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])


def _no_three_collinear(pts):
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear3(pts[i], pts[j], pts[k]):
                    return False
    return True


def _four_concyclic(p1, p2, p3, p4, D):
    """
    Four points (x_i, y_i) in the metric space with distance^2 = dx^2 + D*dy^2
    are concyclic iff they lie on a conic x^2 + D*y^2 + Ax + By + C = 0.
    Check via 4x4 determinant.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    a1 = x1 * x1 + D * y1 * y1
    a2 = x2 * x2 + D * y2 * y2
    a3 = x3 * x3 + D * y3 * y3
    a4 = x4 * x4 + D * y4 * y4
    # det of [[x1,y1,a1,1],[x2,y2,a2,1],[x3,y3,a3,1],[x4,y4,a4,1]]
    det = (y2 * a3 * x4 - y2 * a4 * x3 - a2 * y3 * x4 + a2 * y4 * x3 + x2 * a3 * y4 - x2 * a4 * y3) \
        - (y1 * a3 * x4 - y1 * a4 * x3 - a1 * y3 * x4 + a1 * y4 * x3 + x1 * a3 * y4 - x1 * a4 * y3) \
        + (y1 * a2 * x4 - y1 * a4 * x2 - a1 * y2 * x4 + a1 * y4 * x2 + x1 * a2 * y4 - x1 * a4 * y2) \
        - (y1 * a2 * x3 - y1 * a3 * x2 - a1 * y2 * x3 + a1 * y3 * x2 + x1 * a2 * y3 - x1 * a3 * y2)
    return det == 0


def _no_four_concyclic(pts, D):
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if _four_concyclic(pts[i], pts[j], pts[k], pts[l], D):
                        return False
    return True


def _check_valid(pts, D):
    n = len(pts)
    if n < 3:
        return False
    seen = set()
    for p in pts:
        if p in seen:
            return False
        seen.add(p)
    for i in range(n):
        for j in range(i + 1, n):
            if not _sq_dist_rational(pts[i], pts[j], D):
                return False
    if not _no_three_collinear(pts):
        return False
    if n >= 4 and not _no_four_concyclic(pts, D):
        return False
    return True


def _check_partial(pts, D):
    """Pairwise + collinear only (faster, no concyclic)."""
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            if not _sq_dist_rational(pts[i], pts[j], D):
                return False
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear3(pts[i], pts[j], pts[k]):
                    return False
    return True


def _to_output(pts):
    return [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in pts]


def _gen_circle_candidates(D, max_t=80):
    """
    Generate integer lattice points (x, y) such that x^2 + D*y^2 is a perfect square.
    These are points at rational distance from origin in the (x, y*sqrt(D)) metric.
    """
    cands = set()
    for y in range(-max_t, max_t + 1):
        for x in range(-max_t, max_t + 1):
            val = x * x + D * y * y
            if val <= 0:
                continue
            if _is_perfect_square(val):
                cands.add((x, y))
    return list(cands)


def _gen_param_candidates(D, max_pq=60, max_r=15):
    """
    Generate points via rational parametrization of the unit circle in Q(sqrt(D)):
    (x, y) = ((p^2 - D*q^2)*r, 2*p*q*r) / (p^2 + D*q^2)
    scaled to integer coordinates by multiplying numerator and denominator.
    We produce (Fraction(x), Fraction(y)) pairs.
    """
    cands = set()
    for p in range(1, max_pq + 1):
        for q in range(1, max_pq + 1):
            if gcd(p, q) != 1:
                continue
            denom = p * p + D * q * q
            for r in range(1, max_r + 1):
                x_num = (p * p - D * q * q) * r
                y_num = 2 * p * q * r
                for sx in (1, -1):
                    for sy in (1, -1):
                        pt = (Fraction(sx * x_num, denom), Fraction(sy * y_num, denom))
                        cands.add(pt)
    return list(cands)


def _precompute_compat(cands, D):
    """Precompute pairwise rational-distance compatibility."""
    n = len(cands)
    compat = [[False] * n for _ in range(n)]
    for i in range(n):
        xi, yi = cands[i]
        for j in range(i + 1, n):
            xj, yj = cands[j]
            dx = xi - xj
            dy = yi - yj
            d2 = dx * dx + D * dy * dy
            if d2 < 0:
                continue
            if _is_perfect_square(d2.numerator) and _is_perfect_square(d2.denominator):
                compat[i][j] = compat[j][i] = True
    return compat


def _backtrack(cands, D, target, deadline, compat, max_nodes=300000):
    """
    Backtracking search for a subset of size >= target with all constraints.
    Uses compat matrix for fast pairwise check.
    """
    n = len(cands)
    result = [None]
    nodes = [0]

    def bt(cur, cur_set, idx):
        if time.time() > deadline or nodes[0] > max_nodes:
            return False
        if len(cur) >= target:
            if _no_four_concyclic(cur, D):
                result[0] = list(cur)
                return True
            return False
        for i in range(idx, n):
            nodes[0] += 1
            if time.time() > deadline or nodes[0] > max_nodes:
                return False
            ok = True
            for j in cur_set:
                if not compat[i][j]:
                    ok = False
                    break
            if not ok:
                continue
            new_cur = cur + [cands[i]]
            if _no_three_collinear(new_cur):
                if bt(new_cur, cur_set | {i}, i + 1):
                    return True
        return False

    if bt([], set(), 0):
        return result[0]
    return None


def _greedy_extend(pts, D, cands, deadline, max_size=12):
    """Try to add candidates one by one to pts."""
    best = list(pts)
    for cand in cands:
        if time.time() > deadline:
            break
        if cand in best:
            continue
        new_pts = best + [cand]
        if len(new_pts) <= max_size and _check_valid(new_pts, D):
            best = new_pts
    return best


def _local_search(pts, D, cands, deadline, max_iter=20000):
    """Local search: repeatedly try to add a candidate; if stuck, stop."""
    best = list(pts)
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        added = False
        for cand in cands:
            if cand in best:
                continue
            new_pts = best + [cand]
            if _check_valid(new_pts, D):
                best = new_pts
                added = True
                break
        if not added:
            break
    return best


def _try_hexagon_D3(deadline):
    """
    Attempt to construct a 6-point set in Q(sqrt(3)) inspired by regular hexagon geometry.
    Regular hexagon vertices in complex plane: e^{ik*pi/3} for k=0..5.
    In Q(sqrt(3)), we can represent these as (x, y) with y*sqrt(3).
    Try specific rational coordinates and verify.
    """
    # Regular hexagon of "radius" 2 in the metric: vertices at angles 0, 60, 120, 180, 240, 300
    # In (x, y*sqrt(3)) coordinates:
    # angle 0:   (2, 0)
    # angle 60:  (1, 1)  because 1^2 + 3*1^2 = 4
    # angle 120: (-1, 1)
    # angle 180: (-2, 0)
    # angle 240: (-1, -1)
    # angle 300: (1, -1)
    D = 3
    pts = [
        (Fraction(2), Fraction(0)),
        (Fraction(1), Fraction(1)),
        (Fraction(-1), Fraction(1)),
        (Fraction(-2), Fraction(0)),
        (Fraction(-1), Fraction(-1)),
        (Fraction(1), Fraction(-1)),
    ]
    if _check_valid(pts, D):
        return pts
    # Try scaling or other hexagon-like configurations
    for scale in [1, 2, 3]:
        pts_s = [(x * scale, y * scale) for x, y in pts]
        if _check_valid(pts_s, D):
            return pts_s
    # Try a different hexagon: centered at origin with different radius
    # Vertices at (a, b) where a^2 + 3b^2 = r^2
    for a in range(1, 15):
        for b in range(1, 15):
            r2 = a * a + 3 * b * b
            if _is_perfect_square(r2):
                r = isqrt(r2)
                # Check if we can form 6 points: (±a, ±b), (±b, ±a/sqrt(3))... 
                # This is tricky; try a simpler approach:
                # Use points (a, b), (a, -b), (-a, b), (-a, -b) and two more
                base = [
                    (Fraction(a), Fraction(b)),
                    (Fraction(a), Fraction(-b)),
                    (Fraction(-a), Fraction(b)),
                    (Fraction(-a), Fraction(-b)),
                ]
                # Try adding (0, c) and (0, -c) where c^2*3 = r^2 - a^2... 
                # Actually just check if the 4-point set is valid, then extend
                if _check_valid(base, D):
                    # Try to extend with more points
                    cands = _gen_circle_candidates(D, 20)
                    cands_frac = [(Fraction(x), Fraction(y)) for x, y in cands]
                    extended = _greedy_extend(base, D, cands_frac, deadline, max_size=8)
                    if len(extended) > len(base):
                        return extended
                    return base
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # 2s safety margin

    def remaining():
        return deadline - time.time()

    # Fallback: 5-point set
    points_5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    D_5 = 1
    fracs_5 = [(Fraction(x), Fraction(y)) for x, y in points_5]
    best = {'D': D_5, 'points': _to_output(fracs_5), 'n': 5}

    # Strategy 1: Check known configurations
    known_configs = [
        (1, [(0, 0), (3, 0), (0, 4), (4, 3), (-3, 4), (0, -5)]),
        (1, [(0, 0), (5, 0), (0, 12), (12, 5), (-5, 12), (0, -13)]),
        (1, [(0, 0), (7, 0), (0, 24), (24, 7), (-7, 24), (0, -25)]),
        (1, [(0, 0), (9, 0), (0, 40), (40, 9), (-9, 40), (0, -41)]),
        (1, [(0, 0), (4, 0), (0, 3), (4, 3), (-4, 3), (0, -5)]),
        (1, [(0, 0), (5, 0), (0, 4), (5, 4), (-5, 4), (0, -3)]),
        (1, [(0, 0), (8, 0), (0, 15), (15, 8), (-8, 15), (0, -16)]),
        (1, [(0, 0), (12, 0), (0, 35), (35, 12), (-12, 35), (0, -36)]),
        (2, [(0, 0), (3, 0), (0, 2), (3, 2), (-3, 2), (3, -2)]),
        (2, [(0, 0), (6, 0), (0, 4), (6, 4), (-6, 4), (6, -4)]),
        (3, [(0, 0), (2, 0), (0, 1), (2, 1), (-2, 1), (2, -1)]),
        (3, [(0, 0), (4, 0), (0, 2), (4, 2), (-4, 2), (4, -2)]),
        (5, [(0, 0), (2, 0), (0, 1), (2, 1), (-2, 1), (2, -1)]),
        (7, [(0, 0), (1, 0), (0, 1), (1, 1), (-1, 1), (1, -1)]),
        (10, [(0, 0), (3, 0), (0, 1), (3, 1), (-3, 1), (3, -1)]),
        (13, [(0, 0), (2, 0), (0, 1), (2, 1), (-2, 1), (2, -1)]),
    ]
    for D, pts in known_configs:
        if remaining() < 1:
            break
        if not _is_squarefree(D):
            continue
        fracs = [(Fraction(x), Fraction(y)) for x, y in pts]
        if _check_valid(fracs, D) and len(fracs) > best['n']:
            best = {'D': D, 'points': _to_output(fracs), 'n': len(fracs)}

    # Strategy 2: Try hexagon-inspired construction for D=3
    if remaining() > 5:
        hex_pts = _try_hexagon_D3(deadline)
        if hex_pts and len(hex_pts) > best['n']:
            best = {'D': 3, 'points': _to_output(hex_pts), 'n': len(hex_pts)}

    # Strategy 3: Backtracking search on integer grid for D in [1,2,3,5,7,10,13]
    for D in [1, 2, 3, 5, 7, 10, 13, 17, 19]:
        if remaining() < 8:
            break
        if not _is_squarefree(D):
            continue

        for limit in [5, 6, 7, 8]:
            if remaining() < 5:
                break
            cands_int = [(x, y) for x in range(-limit, limit + 1) for y in range(-limit, limit + 1)]
            cands = [(Fraction(x), Fraction(y)) for x, y in cands_int]
            random.seed(D * 1000 + limit)
            random.shuffle(cands)

            if len(cands) > 500:
                cands = cands[:500]

            compat = _precompute_compat(cands, D)
            target = 6
            max_nodes = 150000 if limit <= 6 else 80000
            result = _backtrack(cands, D, target, deadline, compat, max_nodes=max_nodes)
            if result and len(result) > best['n']:
                best = {'D': D, 'points': _to_output(result), 'n': len(result)}
                # Try to extend
                if remaining() > 5:
                    ext_cands = [(Fraction(x), Fraction(y)) for x in range(-limit - 5, limit + 6)
                                 for y in range(-limit - 5, limit + 6)]
                    extended = _local_search(result, D, ext_cands, deadline, max_iter=15000)
                    if len(extended) > best['n']:
                        best = {'D': D, 'points': _to_output(extended), 'n': len(extended)}
                if best['n'] >= 7:
                    break
        if best['n'] >= 7:
            break

    # Strategy 4: Parametric candidate search
    if remaining() > 8:
        for D in [1, 2, 3, 5, 7, 10, 13]:
            if remaining() < 5:
                break
            if not _is_squarefree(D):
                continue
            cands = _gen_param_candidates(D, max_pq=50, max_r=12)
            random.seed(D * 777)
            random.shuffle(cands)
            if len(cands) > 400:
                cands = cands[:400]
            compat = _precompute_compat(cands, D)
            result = _backtrack(cands, D, 6, deadline, compat, max_nodes=120000)
            if result and len(result) > best['n']:
                best = {'D': D, 'points': _to_output(result), 'n': len(result)}
                if remaining() > 5:
                    extended = _local_search(result, D, cands, deadline, max_iter=10000)
                    if len(extended) > best['n']:
                        best = {'D': D, 'points': _to_output(extended), 'n': len(extended)}

    # Strategy 5: Final extension attempt
    if remaining() > 6 and best['n'] < 8:
        D_cur = best['D']
        pts_cur = [(Fraction(p[0][0], p[0][1]), Fraction(p[1][0], p[1][1])) for p in best['points']]
        limit_ext = 30
        cands_ext = [(Fraction(x), Fraction(y)) for x in range(-limit_ext, limit_ext + 1)
                     for y in range(-limit_ext, limit_ext + 1)]
        extended = _local_search(pts_cur, D_cur, cands_ext, deadline, max_iter=25000)
        if len(extended) > best['n']:
            best = {'D': D_cur, 'points': _to_output(extended), 'n': len(extended)}

    del best['n']
    return best

# EVOLVE-BLOCK-END
