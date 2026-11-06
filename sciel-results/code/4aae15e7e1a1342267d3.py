"""Fresh design: multi-start local search with line-aligned + perturbation moves."""
import time
import random
import math
from collections import Counter

_BUDGET = 135.0  # safety margin below the 150 s wall-clock kill
_DEADLINE = 0.0


def _gcd(a, b):
    a = abs(a); b = abs(b)
    while b:
        a, b = b, a % b
    return a


def _line_key(p1, p2):
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    dx = x2 - x1
    dy = y2 - y1
    g = _gcd(dx, dy)
    if g:
        dx //= g
        dy //= g
    # normalized point on the line
    if dx != 0:
        t = (y1 - dy) // dx
        return (dx, dy, t)
    return (0, 1, x1)


def _score(points):
    """Number of lines containing exactly 4 of the 16 points; 0 if any 5 collinear."""
    keys = [None] * 120
    for i in range(16):
        xi, yi = points[i][0], points[i][1]
        for j in range(i + 1, 16):
            xj, yj = points[j][0], points[j][1]
            dx = xj - xi
            dy = yj - yi
            g = _gcd(dx, dy)
            if g:
                dx //= g
                dy //= g
            if dx != 0:
                t = (yi - dy) // dx
                k = (dx, dy, t)
            else:
                k = (0, 1, xi)
            keys[j - i - 1] = k
    cnt = Counter(keys)
    four = 0
    for c in cnt.values():
        if c >= 10:      # 5 or more points collinear -> invalid
            return 0
        if c == 6:       # exactly 4 points -> C(4,2)=6 pairs
            four += 1
    return four


def _valid(pts):
    for p in pts:
        if not (-1000 <= p[0] <= 1000 and -1000 <= p[1] <= 1000):
            return False
    return len(set(map(tuple, pts))) == 16


def _random_grid():
    while True:
        pts = [[random.randint(-20, 20), random.randint(-20, 20)] for _ in range(16)]
        if len(set(map(tuple, pts))) == 16:
            return pts


def _perturb(pts, rng):
    """Randomly nudge 1-3 points by small offsets (clamped to bounds)."""
    n = min(len(pts), 16)
    m = rng.randint(1, 3)
    for _ in range(m):
        i = rng.randrange(n)
        dx = rng.randint(-3, 3)
        dy = rng.randint(-3, 3)
        if dx == 0 and dy == 0:
            continue
        nx = max(-1000, min(1000, pts[i][0] + dx))
        ny = max(-1000, min(1000, pts[i][1] + dy))
        pts[i] = [nx, ny]
    return pts


def _line_aligned(pts, rng):
    """Move a random point to a random grid cell that lies on an existing line."""
    n = min(len(pts), 16)
    i = rng.randrange(n)
    px, py = pts[i]
    # gather candidate lines through point i (dedup by key)
    seen = set()
    cand = []
    for j in range(n):
        if j == i:
            continue
        k = _line_key(pts[i], pts[j])
        if k in seen:
            continue
        seen.add(k)
        cand.append(k)
    if not cand:
        return _perturb(pts, rng)
    k = rng.choice(cand)
    dx, dy, t = k
    if dx == 0:
        x = t
        if x == px:
            return _perturb(pts, rng)
        y = rng.randint(-20, 20)
        new = [x, y]
    else:
        y = rng.randint(-20, 20)
        if y == py:
            return _perturb(pts, rng)
        num = dx * y + dy * t
        if num % dx != 0:
            return _perturb(pts, rng)
        x = num // dx
        new = [x, y]
    if not (-1000 <= new[0] <= 1000 and -1000 <= new[1] <= 1000):
        return _perturb(pts, rng)
    pts[i] = new
    return pts


def _local_search(pts, rng, deadline):
    n = min(len(pts), 16)
    cur = [list(p) for p in pts]
    cur_score = _score(cur)
    if cur_score == 0:
        return cur, cur_score
    it = 0
    while time.time() < deadline:
        it += 1
        # choose move type
        if rng.random() < 0.5:
            cand = _line_aligned(cur, rng)
        else:
            cand = _perturb(cur, rng)
        if not _valid(cand):
            continue
        s = _score(cand)
        if s > cur_score:
            cur = cand
            cur_score = s
            if cur_score >= 15:
                break
        elif (s == cur_score and rng.random() < 0.15) or (s < cur_score and rng.random() < 0.02):
            cur = cand
            cur_score = s
    return cur, cur_score


def construct():
    global _DEADLINE
    t0 = time.time()
    _DEADLINE = t0 + _BUDGET
    rng = random.Random(20240607)

    best = None
    best_score = -1

    # seed with the 4x4 grid (score 10) and a couple of random starts
    starts = []
    starts.append([[x, y] for x in range(4) for y in range(4)])
    starts.append(_random_grid())
    starts.append(_random_grid())

    for st in starts:
        if time.time() > _DEADLINE:
            break
        p, s = _local_search(st, rng, _DEADLINE)
        if s > best_score:
            best_score = s
            best = p
        if best_score >= 15:
            break

    # keep polishing the best solution with fresh randomness while time remains
    if best is not None and best_score < 15:
        while time.time() < _DEADLINE:
            p, s = _local_search(best, rng, _DEADLINE)
            if s > best_score:
                best_score = s
                best = p
            if best_score >= 15:
                break

    if best is None:
        best = [[x, y] for x in range(4) for y in range(4)]

    # safety: ensure a valid result is returned
    if not _valid(best) or best_score <= 0:
        best = [[x, y] for x in range(4) for y in range(4)]
    return best
