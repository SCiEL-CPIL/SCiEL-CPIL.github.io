import time
from fractions import Fraction
from math import isqrt
from itertools import combinations

_START = time.time()
_DEADLINE = 120.0  # safety margin under 150s kill

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _check_no_three_collinear(pts):
    # pts: list of (x, y) integers or Fractions
    n = len(pts)
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            dx = xj - xi
            dy = yj - yi
            for k in range(j + 1, n):
                xk, yk = pts[k]
                # cross product
                if dx * (yk - yi) - dy * (xk - xi) != 0:
                    continue
                return False
    return True

def _check_no_four_concyclic(pts):
    n = len(pts)
    for (a, b, c, d) in combinations(range(n), 4):
        ax, ay = pts[a]
        bx, by = pts[b]
        cx, cy = pts[c]
        dx, dy = pts[d]
        # circle through a,b,c: general form x^2+y^2 + Dx + Ey + F = 0
        # Solve linear system for D,E,F using a,b,c then check d.
        # Equations: 2*(bx-ax)*D + 2*(by-ay)*E + (bx^2+by^2 - ax^2-ay^2) = 0 ... 
        # Use determinant form (Ptolemy-free): four points concyclic iff
        # the circle through a,b,c also passes through d.
        # Compute using 3x3 determinant (Cayley-Menger style) for concyclicity:
        # | ax^2+ay^2, ax, ay, 1 |
        # | bx^2+by^2, bx, by, 1 | = 0
        # | cx^2+cy^2, cx, cy, 1 |
        # | dx^2+dy^2, dx, dy, 1 |
        a0 = ax * ax + ay * ay
        b0 = bx * bx + by * by
        c0 = cx * cx + cy * cy
        d0 = dx * dx + dy * dy
        # Determinant of 4x4 matrix M where row i = [r_i^2, x_i, y_i, 1]
        M = [
            [a0, ax, ay, 1],
            [b0, bx, by, 1],
            [c0, cx, cy, 1],
            [d0, dx, dy, 1],
        ]
        if _det4(M) == 0:
            return False
    return True

def _det4(M):
    # Laplace expansion along last column (all 1s) to reduce cost
    # det = sum_{i} (-1)^{i+3} * 1 * det3(M without row i, col 3)
    def det3(A):
        return (A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
              - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
              + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]))
    total = 0
    for i in range(4):
        sub = []
        for r in range(4):
            if r == i:
                continue
            sub.append([M[r][0], M[r][1], M[r][2]])
        sign = 1 if (i + 3) % 2 == 0 else -1
        total += sign * det3(sub)
    return total

def _all_distances_rational(pts, D):
    # pts: list of (x, y) with coordinates rational; actual point = (x, y*sqrt(D))
    n = len(pts)
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            dx = xj - xi
            dy = yj - yi
            # dist^2 = dx^2 + D * dy^2
            val = dx * dx + D * dy * dy
            if val < 0:
                return False
            if not _is_square(val):
                return False
    return True

def _build_candidate_pool(max_coord):
    # candidate integer points in a box; we will filter by integer-distance graph density later
    pts = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            pts.append((x, y))
    return pts

def _validate_full(pts, D=1):
    if len(pts) < 3:
        return False
    # distinct
    s = set(pts)
    if len(s) != len(pts):
        return False
    if not _all_distances_rational(pts, D):
        return False
    if not _check_no_three_collinear(pts):
        return False
    if not _check_no_four_concyclic(pts):
        return False
    return True

def _format(pts, D):
    out = []
    for (x, y) in pts:
        # x, y are integers (or Fractions). Represent as [num, den]
        if isinstance(x, Fraction):
            xnum, xden = x.numerator, x.denominator
        else:
            xnum, xden = int(x), 1
        if isinstance(y, Fraction):
            ynum, yden = y.numerator, y.denominator
        else:
            ynum, yden = int(y), 1
        out.append([[xnum, xden], [ynum, yden]])
    return {'D': D, 'points': out}

def _try_extend(seed, candidates, D, max_size, time_budget):
    # greedy with lookahead: at each step pick candidate that (a) keeps validity,
    # (b) maximizes number of remaining candidates that can still be added (flexibility)
    current = list(seed)
    remaining = [c for c in candidates if c not in current]
    while len(current) < max_size and (time.time() - _START) < time_budget:
        best_cand = None
        best_score = -1
        for cand in remaining:
            trial = current + [cand]
            if not _validate_full(trial, D):
                continue
            # flexibility: count of other remaining candidates that are mutually valid with trial
            flex = 0
            for other in remaining:
                if other == cand:
                    continue
                # quick check: cand-other distance rational?
                if not _is_square((other[0]-cand[0])**2 + D*(other[1]-cand[1])**2):
                    continue
                # check collinearity with pairs in current
                ok = True
                for (p, q) in combinations(current, 2):
                    if _collinear(p, q, other):
                        ok = False
                        break
                if not ok:
                    continue
                # check concyclicity with triples in current
                for (p, q, r) in combinations(current, 3):
                    if _concyclic(p, q, r, other):
                        ok = False
                        break
                if ok:
                    flex += 1
            # score: prioritize flexibility, tie-break by distance to centroid (compactness)
            if flex > best_score:
                best_score = flex
                best_cand = cand
        if best_cand is None:
            break
        current.append(best_cand)
        remaining.remove(best_cand)
    return current

def _collinear(p, q, r):
    px, py = p
    qx, qy = q
    rx, ry = r
    return (qx - px) * (ry - py) - (qy - py) * (rx - px) == 0

def _concyclic(p, q, r, s):
    M = []
    for (x, y) in (p, q, r, s):
        M.append([x * x + y * y, x, y, 1])
    return _det4(M) == 0

def construct():
    global _START
    _START = time.time()
    deadline = 120.0  # wall-clock safety margin

    # Start from the known 5-point configuration
    seed = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    D = 1

    # Build a candidate pool around the seed's bounding region, slightly larger
    # Seed spans x in [-56,16], y in [0,63]. Use a box that includes these plus margin.
    minx = min(p[0] for p in seed) - 40
    maxx = max(p[0] for p in seed) + 40
    miny = min(p[1] for p in seed) - 40
    maxy = max(p[1] for p in seed) + 40

    # Pre-filter candidates: must have integer distance to every seed point
    candidates = []
    for x in range(minx, maxx + 1):
        for y in range(miny, maxy + 1):
            cand = (x, y)
            if cand in seed:
                continue
            ok = True
            for (sx, sy) in seed:
                d2 = (x - sx) ** 2 + D * (y - sy) ** 2
                if not _is_square(d2):
                    ok = False
                    break
            if ok:
                candidates.append(cand)

    # Greedy extension
    best = list(seed)
    if (time.time() - _START) < deadline - 5:
        extended = _try_extend(seed, candidates, D, max_size=12, time_budget=deadline - 5)
        if len(extended) > len(best):
            best = extended

    # If we couldn't extend, try a broader but cheaper local search:
    # attempt to add one more point from a larger pool, validating fully.
    if len(best) < 6 and (time.time() - _START) < deadline - 10:
        # larger pool
        minx2 = minx - 60
        maxx2 = maxx + 60
        miny2 = miny - 60
        maxy2 = maxy + 60
        extra = []
        for x in range(minx2, maxx2 + 1):
            for y in range(miny2, maxy2 + 1):
                cand = (x, y)
                if cand in best:
                    continue
                ok = True
                for (sx, sy) in best:
                    d2 = (x - sx) ** 2 + D * (y - sy) ** 2
                    if not _is_square(d2):
                        ok = False
                        break
                if not ok:
                    continue
                extra.append(cand)
                if (time.time() - _START) > deadline - 10:
                    break
            if (time.time() - _START) > deadline - 10:
                break
        for cand in extra:
            trial = best + [cand]
            if _validate_full(trial, D):
                best = trial
                break

    # Final safety: ensure validity
    if not _validate_full(best, D):
        # fall back to the guaranteed 5-point set
        best = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        D = 1

    return _format(best, D)
