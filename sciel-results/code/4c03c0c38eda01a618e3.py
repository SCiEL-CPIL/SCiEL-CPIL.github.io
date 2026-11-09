"""Rewritten program: random search + exact branch-and-bound scorer with deadline guard."""
import random
import time
from math import gcd

# Hard wall-clock budget (evaluator kills at 150s); keep a safety margin.
_DEADLINE = 145.0


def _norm(dx, dy):
    g = gcd(abs(dx), abs(dy))
    if g == 0:
        return (0, 0)
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    return (dx, dy)


def _has_four_collinear(pts):
    n = len(pts)
    for i in range(n):
        xi, yi = pts[i]
        seen = {}
        for j in range(i + 1, n):
            dx = pts[j][0] - xi
            dy = pts[j][1] - yi
            key = _norm(dx, dy)
            c = seen.get(key, 0) + 1
            if c >= 3:
                return True
            seen[key] = c
    return False


def _max_no_three_collinear(pts):
    """Exact largest subset with no three collinear (branch and bound)."""
    n = len(pts)
    if n <= 2:
        return n
    if n <= 4:
        return 3 if _has_four_collinear(pts) else 4

    # For each point, count how many pairs (with it) form a collinear triple.
    pair_count = [0] * n
    for i in range(n):
        xi, yi = pts[i]
        seen = {}
        for j in range(i + 1, n):
            dx = pts[j][0] - xi
            dy = pts[j][1] - yi
            key = _norm(dx, dy)
            seen[key] = seen.get(key, 0) + 1
        for c in seen.values():
            if c >= 2:
                pair_count[i] += c * (c - 1) // 2

    # Upper bound: from a candidate set S, we can add point p only if
    # p creates no collinear triple with two points in S.
    # Precompute, for each point p, the set of pairs (a,b) that would be
    # forbidden (i.e. a, b, p collinear). We bound the number of points
    # still addable by counting, for each p, how many of its forbidden
    # pairs are already both in S.
    forbidden = [set() for _ in range(n)]
    for i in range(n):
        xi, yi = pts[i]
        by_dir = {}
        for j in range(n):
            if j == i:
                continue
            dx = pts[j][0] - xi
            dy = pts[j][1] - yi
            by_dir.setdefault(_norm(dx, dy), []).append(j)
        for lst in by_dir.values():
            L = len(lst)
            if L >= 2:
                for a in range(L):
                    for b in range(a + 1, L):
                        forbidden[i].add((lst[a], lst[b]))

    order = list(range(n))
    order.sort(key=lambda p: pair_count[p], reverse=True)

    best = 0
    cur = [False] * n
    cur_set = set()
    cur_size = 0

    # Greedy initial solution: add points in order if they don't break 3-collinear.
    def _greedy():
        s = set()
        res = []
        for p in order:
            ok = True
            for q in res:
                for r in res:
                    if q == r:
                        continue
                # check triple p,q,r collinear
            # simpler: check against all pairs in s
            ok = True
            lst = list(s)
            for a in range(len(lst)):
                for b in range(a + 1, len(lst)):
                    if (lst[a], lst[b]) in forbidden[p]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                s.add(p)
                res.append(p)
        return len(res), s

    gsize, gset = _greedy()
    best = gsize

    # Branch and bound (depth-first), exploring order.
    def _bound(idx, size):
        # optimistic bound: assume we can add all remaining points
        return size + (n - idx)

    def _dfs(idx, size, s):
        nonlocal best
        if size + (n - idx) <= best:
            return
        if idx == n:
            if size > best:
                best = size
            return
        p = order[idx]
        # Try including p if it doesn't create a collinear triple with two in s.
        ok = True
        lst = list(s)
        for a in range(len(lst)):
            for b in range(a + 1, len(lst)):
                if (lst[a], lst[b]) in forbidden[p]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            s.add(p)
            _dfs(idx + 1, size + 1, s)
            s.remove(p)
        # Try excluding p.
        _dfs(idx + 1, size, s)

    _dfs(0, 0, set())
    return best


def construct():
    t0 = time.time()
    rng = random.Random(1234567)

    # Keep a small pool of candidate coordinate values to encourage variety.
    base_pool = list(range(-40, 41))

    def _random_set():
        pts = set()
        while len(pts) < 20:
            x = rng.choice(base_pool)
            y = rng.choice(base_pool)
            pts.add((x, y))
        return [list(p) for p in pts]

    best_score = None
    best_pts = None

    # Try many random configurations, scoring each with exact B&B.
    while time.time() < _DEADLINE - 5.0:
        cand = _random_set()
        if _has_four_collinear(cand):
            continue
        s = _max_no_three_collinear(cand)
        if best_score is None or s < best_score:
            best_score = s
            best_pts = cand
            # A good (small) score; keep searching a bit more for even smaller.
            if best_score <= 10:
                # tighten: keep going until deadline
                pass

    if best_pts is None:
        # Fallback: return a known-valid configuration (parabola) if search found nothing.
        best_pts = [[x, x * x] for x in range(-10, 10)]

    return best_pts
