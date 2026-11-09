"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

_START = time.time()
_DEADLINE = _START + 110.0  # safety margin under 120s wall-clock


def _line_key(p, q):
    """Canonical integer key for the line through p and q (no floats)."""
    x1, y1 = p
    x2, y2 = q
    dx = x2 - x1
    dy = y2 - y1
    g = gcd(abs(dx), abs(dy))
    if g:
        dx //= g
        dy //= g
    # normalize direction sign
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # a = -dy, b = dx, c = -(a*x1 + b*y1)
    a = -dy
    b = dx
    c = -(a * x1 + b * y1)
    # normalize (a,b,c) by gcd and sign so a>0 or (a==0 and b>0)
    g2 = gcd(gcd(abs(a), abs(b)), abs(c))
    if g2:
        a //= g2
        b //= g2
        c //= g2
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)


def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(points[i], points[j])
            lines[k] = lines.get(k, 0) + 1
    ordinary = 0
    for cnt in lines.values():
        if cnt == 1:
            ordinary += 1
    return ordinary


def _scale(points, s):
    return [[x * s, y * s] for x, y in points]


def _is_valid(points, limit=10**9):
    if len(points) != 12:
        return False
    if len(set(map(tuple, points))) != 12:
        return False
    for x, y in points:
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > limit or abs(y) > limit:
            return False
    # not all collinear
    p0, p1, p2 = points[0], points[1], points[2]
    if (p1[0] - p0[0]) * (p2[1] - p0[1]) == (p1[1] - p0[1]) * (p2[0] - p0[0]):
        # could still be non-collinear if these three are collinear but others not;
        # do a full check
        pass
    for i in range(12):
        if not ((points[i][0] - p0[0]) * (p1[1] - p0[1]) ==
                (points[i][1] - p0[1]) * (p1[0] - p0[0])):
            return True
    return False


def _base_candidates():
    """Yield structured 12-point candidate configurations (integer)."""
    cands = []

    # 3 x 4 grid
    cands.append([[x, y] for x in range(4) for y in range(3)])

    # 4 x 3 grid
    cands.append([[x, y] for x in range(3) for y in range(4)])

    # 3x4 grid with one corner shifted to break symmetry
    g = [[x, y] for x in range(4) for y in range(3)]
    g[0][0] = 5
    cands.append(g)

    # 2 x 6 grid
    cands.append([[x, y] for x in range(6) for y in range(2)])

    # 6 x 2 grid
    cands.append([[x, y] for x in range(2) for y in range(6)])

    # 2 x 3 x 2 block (x in {0,1}, y in {0,1,2}, z in {0,1} mapped to 2D via y+2z)
    cands.append([[x, y + 2 * z] for x in range(2) for y in range(3) for z in range(2)])

    # A "Hessian-like" small set: points on a conic-free arrangement
    # Use points from a 3x3 grid plus 3 extra carefully placed
    base33 = [[x, y] for x in range(3) for y in range(3)]
    extras = [[3, 0], [0, 3], [3, 3]]
    cands.append(base33 + extras)

    # 3x3 grid + (1,3),(3,1),(2,3)
    extras2 = [[1, 3], [3, 1], [2, 3]]
    cands.append(base33 + extras2)

    # 3x3 grid + (0,3),(3,0),(3,3)
    extras3 = [[0, 3], [3, 0], [3, 3]]
    cands.append(base33 + extras3)

    # 3x3 grid + (1,3),(3,1),(3,3)
    extras4 = [[1, 3], [3, 1], [3, 3]]
    cands.append(base33 + extras4)

    # 4x3 with one point moved
    g2 = [[x, y] for x in range(4) for y in range(3)]
    g2[3][0] = 5
    cands.append(g2)

    # 4x3 with two points moved
    g3 = [[x, y] for x in range(4) for y in range(3)]
    g3[0][0] = 5
    g3[3][2] = 5
    cands.append(g3)

    # 5x? arrangements
    cands.append([[x, y] for x in range(5) for y in range(3)][:12])

    return cands


def _local_search(points, max_iters=20000):
    """Hill-climb: try moving a single point to a nearby integer location."""
    import random
    rng = random.Random(12345)
    cur = [list(p) for p in points]
    cur_score = _count_ordinary(cur)
    best = [list(p) for p in cur]
    best_score = cur_score

    for it in range(max_iters):
        if time.time() > _DEADLINE - 20.0:
            break
        # pick a random point and a small random displacement
        idx = rng.randrange(12)
        dx = rng.randrange(-3, 4)
        dy = rng.randrange(-3, 4)
        if dx == 0 and dy == 0:
            continue
        cand = [list(p) for p in cur]
        cand[idx][0] += dx
        cand[idx][1] += dy
        if not _is_valid(cand):
            continue
        sc = _count_ordinary(cand)
        if sc < cur_score:
            cur = cand
            cur_score = sc
            if cur_score < best_score:
                best_score = cur_score
                best = [list(p) for p in cur]
            if best_score <= 6:
                break
        elif sc == cur_score and rng.random() < 0.15:
            # occasional plateau move to escape local minima
            cur = cand
    return best, best_score


def construct():
    # Start from structured candidates, keep the best, then local-search.
    best_pts = None
    best_score = 10**9

    for cand in _base_candidates():
        if not _is_valid(cand):
            continue
        sc = _count_ordinary(cand)
        if sc < best_score:
            best_score = sc
            best_pts = [list(p) for p in cand]
        if best_score <= 6:
            break

    # Also try scaled versions to allow integer reach of lower bound if needed
    if best_score > 6:
        for s in (2, 3, 5):
            cand = _scale(_base_candidates()[0], s)
            if _is_valid(cand):
                sc = _count_ordinary(cand)
                if sc < best_score:
                    best_score = sc
                    best_pts = cand

    # Local search to improve
    if best_pts is not None and best_score > 6:
        pts, sc = _local_search(best_pts)
        if sc < best_score:
            best_score = sc
            best_pts = pts

    if best_pts is None:
        # Fallback: the 3x4 grid
        best_pts = [[x, y] for x in range(4) for y in range(3)]

    # Ensure coordinates within bound
    for p in best_pts:
        p[0] = int(p[0])
        p[1] = int(p[1])

    return best_pts
# EVOLVE-BLOCK-END
