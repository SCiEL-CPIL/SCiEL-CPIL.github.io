"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
from collections import Counter, defaultdict

# EVOLVE-BLOCK-START

def _line_key(p1, p2):
    """Canonical key for the line through p1 and p2 as (A,B,C) with A*x+B*y=C, gcd-normalized, sign-fixed."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = math.gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    A = dy
    B = -dx
    C = A * x1 + B * y1
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)


def _score(points):
    """Return (exact4, has_five). Uses pair-counting: a line with k points yields C(k,2) pairs."""
    pair_counts = Counter()
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                pair_counts[key] += 1
    exact4 = 0
    has_five = False
    for cnt in pair_counts.values():
        disc = 1 + 8 * cnt
        s = math.isqrt(disc)
        if s * s != disc:
            # Should not happen for a valid configuration; skip
            continue
        k = (1 + s) // 2
        if k == 4:
            exact4 += 1
        elif k >= 5:
            has_five = True
    return exact4, has_five


def _affine_image(base, a, b, c, d, e, f):
    """Apply x' = a*x + b*y + e, y' = c*x + d*y + f to each point of base."""
    out = []
    for (x, y) in base:
        out.append([a * x + b * y + e, c * x + d * y + f])
    return out


# Small base patterns (each is a set of 4 integer points, kept near origin).
_BASES = [
    [(0, 0), (1, 0), (0, 1), (1, 1)],   # 2x2 square
    [(0, 0), (1, 0), (2, 0), (3, 0)],   # 4 in a row
    [(0, 0), (1, 0), (0, 1), (2, 1)],   # parallelogram
    [(0, 0), (1, 0), (1, 1), (2, 0)],   # zigzag
    [(0, 0), (1, 0), (2, 0), (1, 1)],   # trapezoid
    [(0, 0), (1, 0), (0, 1), (-1, 1)],  # slanted
    [(0, 0), (1, 0), (2, 0), (0, 2)],   # L-ish
    [(0, 0), (1, 0), (0, 1), (1, 1)],   # square (dup, different index)
    [(0, 0), (1, 0), (2, 0), (3, 0)],   # line (dup)
    [(0, 0), (1, 0), (0, 1), (1, 2)],   # slanted quad
    [(0, 0), (1, 1), (2, 0), (1, -1)],  # diamond
    [(0, 0), (1, 0), (2, 1), (3, 0)],   # wave
]


def _in_bounds(pts):
    for (x, y) in pts:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
    return True


def _distinct(pts):
    return len(set(tuple(p) for p in pts)) == len(pts)


def _random_affine(rng):
    """A random small affine map (integer coefficients, small range)."""
    a = rng.randint(-2, 2)
    b = rng.randint(-2, 2)
    c = rng.randint(-2, 2)
    d = rng.randint(-2, 2)
    e = rng.randint(-40, 40)
    f = rng.randint(-40, 40)
    return (a, b, c, d, e, f)


def _random_block(rng):
    base = rng.choice(_BASES)
    m = _random_affine(rng)
    return _affine_image(base, *m)


def _random_config(rng):
    pts = []
    for _ in range(4):
        pts.extend(_random_block(rng))
    return pts


def _local_improve(points, deadline, rng):
    """Greedy single-point local search with simulated annealing acceptance."""
    pts = [list(p) for p in points]
    exact4, has_five = _score(pts)
    if has_five:
        # try to fix by randomizing a few points
        for _ in range(200):
            i = rng.randrange(16)
            pts[i] = [rng.randint(-30, 30), rng.randint(-30, 30)]
            if _distinct(pts) and _in_bounds(pts):
                e2, h2 = _score(pts)
                if not h2:
                    exact4 = e2
                    break
        else:
            return pts, exact4 if not has_five else 0
    best = [list(p) for p in pts]
    best_score = exact4
    T = 3.0
    n_iter = 0
    while time.time() < deadline and T > 0.05:
        i = rng.randrange(16)
        ox, oy = pts[i]
        dx = rng.randint(-6, 6)
        dy = rng.randint(-6, 6)
        if dx == 0 and dy == 0:
            continue
        nx, ny = ox + dx, oy + dy
        if abs(nx) > 1000 or abs(ny) > 1000:
            continue
        if any(pts[j] == [nx, ny] for j in range(16)):
            continue
        pts[i] = [nx, ny]
        e2, h2 = _score(pts)
        if h2:
            # invalid: reject
            pts[i] = [ox, oy]
            continue
        delta = e2 - exact4
        if delta >= 0 or rng.random() < math.exp(delta / T):
            exact4 = e2
            if e2 > best_score:
                best_score = e2
                best = [list(p) for p in pts]
        else:
            pts[i] = [ox, oy]
        n_iter += 1
        if n_iter % 200 == 0:
            T *= 0.97
    return best, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 150s

    rng = random.Random(12345)

    # Seed: 4x4 grid (known to give many exact-4 lines)
    grid = [[x, y] for x in range(4) for y in range(4)]
    e, h = _score(grid)
    best_score = e if not h else 0
    best_points = [list(p) for p in grid]

    # Phase 1: try many random 4-block affine configurations, refine the good ones.
    budget_per_refine = 4.0
    tries = 0
    while time.time() < deadline - 30.0:
        cfg = _random_config(rng)
        if not _distinct(cfg) or not _in_bounds(cfg):
            tries += 1
            continue
        e, h = _score(cfg)
        if h:
            tries += 1
            continue
        # If promising, refine locally with a sub-budget
        if e >= best_score - 2:
            sub_deadline = min(deadline - 25.0, time.time() + budget_per_refine)
            refined, rscore = _local_improve(cfg, sub_deadline, rng)
            if rscore > best_score:
                best_score = rscore
                best_points = [list(p) for p in refined]
        tries += 1
        if tries > 4000:
            break

    # Phase 2: refine the current best with the remaining budget.
    sub_deadline = deadline - 2.0
    refined, rscore = _local_improve(best_points, sub_deadline, rng)
    if rscore > best_score:
        best_score = rscore
        best_points = [list(p) for p in refined]

    # Final safety check
    e, h = _score(best_points)
    if h:
        # Fallback to grid
        best_points = [list(p) for p in grid]

    return [list(p) for p in best_points]

# EVOLVE-BLOCK-END
