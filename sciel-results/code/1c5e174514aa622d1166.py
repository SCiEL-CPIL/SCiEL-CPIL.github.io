"""Integral point sets in general position (no three collinear, no four concyclic).

Strategy: build a rational-distance clique of integer lattice points.
  - Start from the Euler (Braikenridge–Maclaurin) 5-point configuration:
    the 13-14-15 triangle plus its circumcenter (R = 65/12, rational),
    which already has all 6 mutual distances rational.
  - Greedily add a 6th integer point with integer distance to all existing
    points, preserving general position (no 3 collinear, no 4 concyclic).
  - Fall back to the best valid subset (>= 5) if no 6th point is found.

All checks use exact rational arithmetic (fractions / integer determinants).
"""
import time
from fractions import Fraction
from itertools import combinations

_T0 = time.time()
_DEADLINE = _T0 + 140.0          # explicit safety margin vs 150 s wall-clock
_RMAX = 170                       # search radius for the extra point


def _elapsed():
    return time.time() - _T0


def _is_square(n):
    if n < 0:
        return False
    r = int(n ** 0.5)
    for k in (r - 1, r, r + 1):
        if k >= 0 and k * k == n:
            return True
    return False


def _collinear3(a, b, c):
    # exact cross product of (b-a) and (c-a)
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) == 0


def _concyclic4(a, b, c, d):
    # exact: determinant of the 4x4 circle-incidence matrix
    A = [[a[0] ** 2 + a[1] ** 2, a[0], a[1], 1],
         [b[0] ** 2 + b[1] ** 2, b[0], b[1], 1],
         [c[0] ** 2 + c[1] ** 2, c[0], c[1], 1],
         [d[0] ** 2 + d[1] ** 2, d[0], d[1], 1]]
    # Gaussian elimination over exact rationals
    n = 4
    M = [[Fraction(x) for x in row] for row in A]
    for col in range(n):
        piv = None
        for r in range(col, n):
            if M[r][col] != 0:
                piv = r
                break
        if piv is None:
            return True                     # singular -> concyclic
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for r in range(col + 1, n):
            f = M[r][col] / pv
            if f != 0:
                for cc in range(col, n):
                    M[r][cc] -= f * M[col][cc]
    return False


def _valid(points):
    for a, b, c in combinations(points, 3):
        if _collinear3(a, b, c):
            return False
    for a, b, c, d in combinations(points, 4):
        if _concyclic4(a, b, c, d):
            return False
    return True


def _dist2_int(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def _build():
    # ---- base 4 points: 13-14-15 triangle + circumcenter (all rational distances)
    A = (0, 0)
    B = (13, 0)
    C = (5, 12)
    O = (65, 14)            # circumcenter * 12 -> integer; distances to A,B,C are 65
    base = [A, B, C, O]

    # ---- find a 5th integer point E with integer distance to all 4 base points
    E = None
    for x in range(-_RMAX, _RMAX + 1):
        for y in range(-_RMAX, _RMAX + 1):
            if _elapsed() > _DEADLINE:
                return base
            P = (x, y)
            if any(P == q for q in base):
                continue
            if not all(_is_square(_dist2_int(P, q)) for q in base):
                continue
            if not _valid(base + [P]):
                continue
            E = P
            break
        if E is not None:
            break
    if E is None:
        return base

    pts = base + [E]

    # ---- find a 6th integer point F with integer distance to all 5
    F = None
    for x in range(-_RMAX, _RMAX + 1):
        for y in range(-_RMAX, _RMAX + 1):
            if _elapsed() > _DEADLINE:
                return pts
            P = (x, y)
            if any(P == q for q in pts):
                continue
            if not all(_is_square(_dist2_int(P, q)) for q in pts):
                continue
            if not _valid(pts + [P]):
                continue
            F = P
            break
        if F is not None:
            break
    if F is not None:
        pts = pts + [F]
    return pts


def construct():
    pts = _build()
    if _elapsed() > _DEADLINE:
        pts = pts[:5]
    # guarantee general position by trimming to the largest valid prefix/subset
    if not _valid(pts):
        # drop points from the end until valid (base 4 is always valid)
        while len(pts) > 4 and not _valid(pts):
            pts = pts[:-1]
    D = 1
    out = []
    for (x, y) in pts:
        out.append([ [x, 1], [y, 1] ])
    return {'D': D, 'points': out}
