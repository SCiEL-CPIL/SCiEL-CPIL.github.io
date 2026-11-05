"""Program using MODEL-AND-OPTIMIZE: local search + geometric verification for integral point sets.

Model:
- Parameterize as (D, list of points with rational x and y*sqrt(D)).
- Objective: maximize number of distinct points n (3..12) such that:
  * all pairwise squared distances are rational (equivalently integral after scaling for a fixed D)
  * no three points are collinear (exact determinant 0 check in affine 2-space)
  * no four points are concyclic (determinant-based cyclic condition using squared distances)
  * |numerators|, |denominators| <= 1e40, D squarefree <= 1e12
- For D=1, reduce to integer lattice with integer squared distances.

Optimization:
- Start from a known 5-point integral set.
- Perform a time-budgeted local search over a bounded integer grid to add a 6th point
  that (i) has integer squared distances to the existing 5 and (ii) preserves all
  collinearity/concyclicity constraints.
- If time remains, do a broader search up to n=8 on a slightly larger grid but keep
  constraints tight to stay within the wall-clock limit.
- Always return a valid configuration of maximum found n (starting from n=5).

Deadline:
- Use time.perf_counter(); stop loops before ~110s wall-clock, then finalize.
"""
import time
from math import isqrt
from itertools import combinations

TIME_LIMIT = 150.0
SAFETY = 30.0
DEADLINE = TIME_LIMIT - SAFETY

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _gcd(a, b):
    a, b = abs(int(a)), abs(int(b))
    while b:
        a, b = b, a % b
    return a

def _sqdist(p, q):
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + dy * dy

def _collinear3(A, B, C):
    # (B.x - A.x)*(C.y - A.y) - (B.y - A.y)*(C.x - A.x)
    return (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])

def _conic_coeff(A, B, C, D):
    # determinant
    # | x1 y1 x1^2+y1^2 1 |
    # | x2 y2 x2^2+y2^2 1 |
    # | x3 y3 x3^2+y3^2 1 |
    # | x4 y4 x4^2+y4^2 1 |
    (x1, y1) = A
    (x2, y2) = B
    (x3, y3) = C
    (x4, y4) = D
    s1 = x1*x1 + y1*y1
    s2 = x2*x2 + y2*y2
    s3 = x3*x3 + y3*y3
    s4 = x4*x4 + y4*y4
    # Expand determinant with column 4 elimination (use row1 as pivot)
    # D = x1*y2*(s3 - s4) - x1*y3*(s2 - s4) + x1*y4*(s2 - s3)
    #   - y1*x2*(s3 - s4) + y1*x3*(s2 - s4) - y1*x4*(s2 - s3)
    #   + (s2*y3*x4 - s2*y4*x3) - (s3*y2*x4 - s3*y4*x2) + (s4*y2*x3 - s4*y3*x2)
    d = (
        x1 * y2 * (s3 - s4) - x1 * y3 * (s2 - s4) + x1 * y4 * (s2 - s3)
        - y1 * x2 * (s3 - s4) + y1 * x3 * (s2 - s4) - y1 * x4 * (s2 - s3)
        + s2 * (y3 * x4 - y4 * x3)
        - s3 * (y2 * x4 - y4 * x2)
        + s4 * (y2 * x3 - y3 * x2)
    )
    return d

def _valid_set(pts):
    # Check distinct
    seen = set(pts)
    if len(seen) != len(pts):
        return False
    # Check all squared distances are perfect squares (D=1)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if not _is_square(_sqdist(pts[i], pts[j])):
                return False
    # No 3 collinear
    for a, b, c in combinations(pts, 3):
        if _collinear3(a, b, c) == 0:
            return False
    # No 4 concyclic
    for a, b, c, d in combinations(pts, 4):
        if _conic_coeff(a, b, c, d) == 0:
            return False
    return True

def _format_points(pts):
    # D=1 so y is rational y with den=1
    return {"D": 1, "points": [[[x, 1], [y, 1]] for (x, y) in pts]}

def _try_extend(current, deadline):
    # current: list of (x, y) integer points
    # try adding a point from a bounded grid; if found, verify full set.
    n = len(current)
    xs = [p[0] for p in current]
    ys = [p[1] for p in current]
    xmin, xmax = min(xs) - 150, max(xs) + 150
    ymin, ymax = min(ys) - 150, max(ys) + 150

    existing = set(current)

    t0 = time.perf_counter()
    cand = None

    # Precompute squared distances from each existing point to candidate
    # Use a smarter search: iterate over possible distances from point 0
    # For each distance d from point 0, find integer solutions to (x-x0)^2 + (y-y0)^2 = d^2
    # This is more efficient than scanning the entire grid

    x0, y0 = current[0]

    # Try various distances from the first point
    max_dist = 200
    for d in range(1, max_dist + 1):
        if time.perf_counter() > deadline:
            break
        d2 = d * d
        # Find all (dx, dy) such that dx^2 + dy^2 = d2
        for dx in range(-d, d + 1):
            dy2 = d2 - dx * dx
            if dy2 < 0:
                continue
            dy = isqrt(dy2)
            if dy * dy != dy2:
                continue
            # Test (dx, dy) and (dx, -dy) if dy > 0
            for dy_sign in [(dx, dy), (dx, -dy)] if dy > 0 else [(dx, dy)]:
                dx_val, dy_val = dy_sign
                x = x0 + dx_val
                y = y0 + dy_val
                if (x, y) in existing:
                    continue
                if x < xmin or x > xmax or y < ymin or y > ymax:
                    continue

                # Check squared distances to all other existing points
                ok = True
                for i in range(1, n):
                    dx_i = x - current[i][0]
                    dy_i = y - current[i][1]
                    if not _is_square(dx_i * dx_i + dy_i * dy_i):
                        ok = False
                        break
                if not ok:
                    continue

                # quick collinearity test vs pairs
                bad = False
                for a in range(n):
                    for b in range(a + 1, n):
                        if _collinear3(current[a], current[b], (x, y)) == 0:
                            bad = True
                            break
                    if bad:
                        break
                if bad:
                    continue

                # quick concyclicity with 3 of the existing (any 3)
                for a, b, c in combinations(range(n), 3):
                    if _conic_coeff(current[a], current[b], current[c], (x, y)) == 0:
                        bad = True
                        break
                if bad:
                    continue

                # If passed local tests, candidate found; stop to do full validation
                cand = (x, y)
                break
            if cand is not None:
                break
        if cand is not None:
            break

    if cand is None:
        # Fallback: try a broader grid search if time remains
        for x in range(xmin, xmax + 1):
            if time.perf_counter() > deadline:
                break
            for y in range(ymin, ymax + 1):
                if (x, y) in existing:
                    continue
                dx2 = (x - x0) ** 2 + (y - y0) ** 2
                if not _is_square(dx2):
                    continue
                ok = True
                for i in range(1, n):
                    dx = x - current[i][0]
                    dy = y - current[i][1]
                    if not _is_square(dx * dx + dy * dy):
                        ok = False
                        break
                if not ok:
                    continue

                bad = False
                for a in range(n):
                    for b in range(a + 1, n):
                        if _collinear3(current[a], current[b], (x, y)) == 0:
                            bad = True
                            break
                    if bad:
                        break
                if bad:
                    continue

                for a, b, c in combinations(range(n), 3):
                    if _conic_coeff(current[a], current[b], current[c], (x, y)) == 0:
                        bad = True
                        break
                if bad:
                    continue

                cand = (x, y)
                break
            if cand is not None:
                break

    if cand is None:
        return current

    new_set = current + [cand]
    if _valid_set(new_set):
        return new_set
    return current

def construct():
    # Seed 5-point set
    base = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]

    # Validate base (should be valid)
    if not _valid_set(base):
        # Fallback to smaller valid known set: use the first 3 points
        base = base[:3]

    t0 = time.perf_counter()
    deadline = t0 + DEADLINE

    best = base[:]

    # Try iterative extension with time guard
    while len(best) < 12 and (time.perf_counter() < deadline):
        extended = _try_extend(best, deadline)
        if len(extended) <= len(best):
            break
        best = extended

    # Final safeguard
    if not _valid_set(best):
        # Ensure we return at least a minimal valid set
        best = base[:3]

    return _format_points(best)
