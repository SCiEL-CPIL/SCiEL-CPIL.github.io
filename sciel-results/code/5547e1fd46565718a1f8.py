"""Rewritten program: dense collinearity construction with no-four constraint.

Goal: 20 distinct integer points, |x|,|y| <= 100, NO four collinear.
Minimize the largest subset with no three collinear (evaluator computes score).
We maximize the number of 3-collinear triples (dense collinearity) subject to
the hard no-four-collinear constraint, via greedy construction + local search.
"""
import time
import random
import itertools

# Hard deadline: return a valid config well before the 150s kill.
_START = time.time()
_DEADLINE = 138.0  # seconds of wall-clock budget

def _elapsed():
    return time.time() - _START

def _remaining():
    return _DEADLINE - _elapsed()

def _cross(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

def _collinear3(p, q, r):
    return _cross(p[0], p[1], q[0], q[1], r[0], r[1]) == 0

def _line_key(p, q):
    """Normalized line (a,b,c) for ax+by=c, gcd-normalized, sign-canonical."""
    x1, y1 = p
    x2, y2 = q
    a = y1 - y2
    b = x2 - x1
    c = -(a * x1 + b * y1)
    # normalize by gcd
    import math
    g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
    if g == 0:
        return (0, 0, 0)
    a //= g; b //= g; c //= g
    # canonical sign: first non-zero of (a,b) positive
    if a < 0 or (a == 0 and b < 0):
        a = -a; b = -b; c = -c
    return (a, b, c)

def _has_four_collinear(pts):
    """Return True if any line contains >= 4 of the given points."""
    n = len(pts)
    seen = set()
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(pts[i], pts[j])
            if k in seen:
                continue
            seen.add(k)
            cnt = 0
            for p in pts:
                if _cross(pts[i][0], pts[i][1], pts[j][0], pts[j][1], p[0], p[1]) == 0:
                    cnt += 1
                    if cnt >= 4:
                        return True
    return False

def _triple_count(pts):
    """Number of 3-collinear triples among pts (O(n^3), n small)."""
    n = len(pts)
    c = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear3(pts[i], pts[j], pts[k]):
                    c += 1
    return c

def _valid(pts):
    """Distinct, in bounds, exactly 20, no four collinear."""
    if len(pts) != 20:
        return False
    if len(set(map(tuple, pts))) != 20:
        return False
    for (x, y) in pts:
        if not (-100 <= x <= 100 and -100 <= y <= 100):
            return False
    if _has_four_collinear(pts):
        return False
    return True

def _greedy_build(pool, rng, target=20):
    """Greedily pick `target` points from pool maximizing new triples, no-four."""
    pts = []
    pool = list(pool)
    rng.shuffle(pool)
    # seed with a few random points
    for p in pool[:3]:
        pts.append(p)
    while len(pts) < target:
        best_p = None
        best_score = -1
        # try candidates; limit work per iteration
        for p in pool:
            if p in pts:
                continue
            cand = pts + [p]
            if _has_four_collinear(cand):
                continue
            s = _triple_count(cand)
            if s > best_score:
                best_score = s
                best_p = p
        if best_p is None:
            # fallback: pick any that keeps no-four
            for p in pool:
                if p in pts:
                    continue
                cand = pts + [p]
                if not _has_four_collinear(cand):
                    best_p = p
                    break
        if best_p is None:
            break
        pts.append(best_p)
    return pts

def _local_search(pts, rng, max_iters=20000):
    """Swap/move local search maximizing triple count under no-four constraint."""
    pts = [list(p) for p in pts]
    n = len(pts)
    cur = _triple_count(pts)
    pool = _candidate_pool()
    best = [list(p) for p in pts]
    best_score = cur
    it = 0
    while it < max_iters and _remaining() > 2.0:
        it += 1
        # random move: replace a random point with a random pool point
        idx = rng.randrange(n)
        newp = pool[rng.randrange(len(pool))]
        if newp == pts[idx]:
            continue
        cand = [list(p) for p in pts]
        cand[idx] = list(newp)
        if len(set(map(tuple, cand))) != n:
            continue
        if _has_four_collinear(cand):
            continue
        s = _triple_count(cand)
        if s > cur:
            cur = s
            pts = cand
            if cur > best_score:
                best_score = cur
                best = [list(p) for p in pts]
        it += 1
    return best, best_score

def _candidate_pool():
    """Dense grid pool within bounds; biases toward small coordinates for collinearity."""
    pool = []
    # a moderately dense grid
    for x in range(-12, 13):
        for y in range(-12, 13):
            pool.append((x, y))
    return pool

def _restart(rng, pool):
    pts = _greedy_build(pool, rng)
    if len(pts) < 20:
        return None
    pts, sc = _local_search(pts, rng, max_iters=4000)
    if _valid(pts):
        return pts, sc
    return None

def construct():
    rng = random.Random(20240517)
    pool = _candidate_pool()

    best_pts = None
    best_score = -1

    # Several restarts with varying local-search budgets, guarded by deadline.
    restart = 0
    while _remaining() > 6.0:
        restart += 1
        # increasing local search budget as time allows
        ls_budget = 3000 + 1500 * restart
        pts = _greedy_build(pool, rng)
        if len(pts) < 20:
            continue
        pts, sc = _local_search(pts, rng, max_iters=ls_budget)
        if _valid(pts) and sc > best_score:
            best_score = sc
            best_pts = [list(p) for p in pts]
        if _remaining() < 6.0:
            break

    # Final safety: if nothing valid, fall back to a known-valid dense-ish config.
    if best_pts is None:
        # Fallback: a configuration guaranteed valid (no four collinear).
        # Use a perturbed 3-row structure; verify, else parabola (valid, score 20).
        fb = [[x, x * x] for x in range(-10, 10)]
        if _valid(fb):
            best_pts = fb
        else:
            # parabola always valid (no three collinear -> no four)
            best_pts = [[x, x * x] for x in range(-10, 10)]

    # Ensure return type matches: list of [x, y]
    return [[int(p[0]), int(p[1])] for p in best_pts]
