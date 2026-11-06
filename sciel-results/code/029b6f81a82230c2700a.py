"""Rewritten program: randomized local search maximizing exact-4-collinear lines."""
import time
import random
import math

# EVOLVE-BLOCK-START
_START_TIME = None
_TIME_LIMIT = 140.0  # safety margin under 150s kill

def _time_left():
    if _START_TIME is None:
        return _TIME_LIMIT
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _line_key(p1, p2):
    """Normalized key for the line through two points."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return None
    g = math.gcd(dx, dy) if (dx != 0 or dy != 0) else 1
    if g < 0:
        g = -g
    dx //= g
    dy //= g
    # canonical direction
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # line offset: a*x + b*y = c where (a,b) is normal (dy, -dx)
    a = dy
    b = -dx
    c = a * p1[0] + b * p1[1]
    g2 = math.gcd(abs(a), abs(b))
    if g2:
        a //= g2
        b //= g2
        c //= g2
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
        c = -c
    return (a, b, c)

def _score(points):
    """Return (num_exact4_lines, valid). valid=False if any 5+ collinear."""
    n = len(points)
    # bucket points by line key
    from collections import defaultdict
    line_pts = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(points[i], points[j])
            if k is None:
                continue
            line_pts[k].append((i, j))
    # For each line key, gather the set of point indices
    line_set = defaultdict(set)
    for k, pairs in line_pts.items():
        for (i, j) in pairs:
            line_set[k].add(i)
            line_set[k].add(j)
    exact4 = 0
    for k, s in line_set.items():
        if len(s) == 4:
            exact4 += 1
        elif len(s) >= 5:
            return exact4, False
    return exact4, True

def _inbounds(p):
    return -1000 <= p[0] <= 1000 and -1000 <= p[1] <= 1000

def _valid_config(points):
    """Check distinct, in bounds, no 5 collinear."""
    if len(points) != 16:
        return False
    seen = set()
    for p in points:
        if not _inbounds(p):
            return False
        if (p[0], p[1]) in seen:
            return False
        seen.add((p[0], p[1]))
    _, ok = _score(points)
    return ok

def _random_point(rng, lo=-1000, hi=1000):
    return [rng.randint(lo, hi), rng.randint(lo, hi)]

def _init_grid(rng):
    """Start from a 4x4 grid with some random jitter/shifts to diversify."""
    base = [[x, y] for x in range(4) for y in range(4)]
    # random translation and small per-point perturbation chance
    tx = rng.randint(-5, 5)
    ty = rng.randint(-5, 5)
    pts = [[p[0] + tx, p[1] + ty] for p in base]
    for i in range(len(pts)):
        if rng.random() < 0.3:
            pts[i] = [pts[i][0] + rng.randint(-3, 3), pts[i][1] + rng.randint(-3, 3)]
    return pts

def _init_random(rng):
    pts = []
    seen = set()
    while len(pts) < 16:
        p = _random_point(rng, -50, 50)
        key = (p[0], p[1])
        if key not in seen:
            seen.add(key)
            pts.append(p)
    return pts

def _swap_mutate(points, rng):
    """Swap two points, or replace one point."""
    pts = [list(p) for p in points]
    if rng.random() < 0.5:
        i, j = rng.sample(range(16), 2)
        pts[i], pts[j] = pts[j], pts[i]
    else:
        i = rng.randint(0, 15)
        # replace with a nearby point to keep structure
        base = points[i]
        newp = [base[0] + rng.randint(-20, 20), base[1] + rng.randint(-20, 20)]
        if not _inbounds(newp):
            newp = _random_point(rng, -100, 100)
        pts[i] = newp
    return pts

def _perturb(points, rng):
    """Move a single point to a random location (bigger jump)."""
    pts = [list(p) for p in points]
    i = rng.randint(0, 15)
    pts[i] = _random_point(rng, -100, 100)
    return pts

def construct():
    global _START_TIME
    _START_TIME = time.time()
    rng = random.Random()

    best_pts = None
    best_score = -1

    # Initial candidate: the 4x4 grid (known ~20 four-point lines)
    grid = [[x, y] for x in range(4) for y in range(4)]
    if _valid_config(grid):
        s, _ = _score(grid)
        best_pts = [list(p) for p in grid]
        best_score = s

    def try_improve(init_pts, iterations):
        nonlocal best_pts, best_score
        if not _valid_config(init_pts):
            return
        cur = [list(p) for p in init_pts]
        cur_score, _ = _score(cur)
        if cur_score > best_score:
            best_score = cur_score
            best_pts = [list(p) for p in cur]
        for _ in range(iterations):
            if _time_left() < 2.0:
                return
            if rng.random() < 0.8:
                cand = _swap_mutate(cur, rng)
            else:
                cand = _perturb(cur, rng)
            if not _valid_config(cand):
                continue
            cs, _ = _score(cand)
            # accept if better, or occasionally accept equal/worse to escape
            if cs > cur_score or (cs == cur_score and rng.random() < 0.1) or (cs < cur_score and rng.random() < 0.02):
                cur = cand
                cur_score = cs
                if cur_score > best_score:
                    best_score = cur_score
                    best_pts = [list(p) for p in cur]

    # Multiple restarts with diverse seeds
    restarts = 0
    while _time_left() > 3.0:
        restarts += 1
        if restarts % 3 == 1:
            init = _init_grid(rng)
        elif restarts % 3 == 2:
            init = _init_random(rng)
        else:
            init = [list(p) for p in best_pts] if best_pts else _init_grid(rng)
        # adaptive iterations based on remaining time
        iters = 200
        try_improve(init, iters)

    if best_pts is None:
        # fallback
        best_pts = [[x, y] for x in range(4) for y in range(4)]

    return best_pts
# EVOLVE-BLOCK-END
