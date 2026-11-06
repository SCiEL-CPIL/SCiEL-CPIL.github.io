"""Rewritten program: structural change to a local-search optimiser over 12-point sets."""
import time
import random
import math
from collections import defaultdict

_DEADLINE = 140  # seconds, safety margin under 150s wall-clock


def _line_key(p, q):
    """Return a canonical integer key for the line through two integer points (p, q)."""
    ax, ay = p
    bx, by = q
    dx = bx - ax
    dy = by - ay
    g = math.gcd(dx, dy)
    dx //= g
    dy //= g
    # Canonicalise direction: lexicographically smallest
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # Normal form: a*x + b*y + c = 0 with (a, b) = (dy, -dx) after canonicalisation
    a = dy
    b = -dx
    c = -(a * ax + b * ay)
    g2 = math.gcd(a, b, c)
    if g2 != 0:
        a //= g2
        b //= g2
        c //= g2
    if a < 0 or (a == 0 and (b < 0 or (b == 0 and c < 0))):
        a, b, c = -a, -b, -c
    return (a, b, c)


def _count_ordinary_lines(pts):
    """Count lines containing exactly two of the points."""
    n = len(pts)
    line_to_count = defaultdict(int)
    for i in range(n):
        for j in range(i + 1, n):
            line_to_count[_line_key(pts[i], pts[j])] += 1
    return sum(1 for c in line_to_count.values() if c == 2)


def _not_all_collinear(pts):
    """True unless all points lie on one line."""
    if len(pts) < 3:
        return len(pts) < 3
    a, b, c = pts[0], pts[1], pts[2]
    cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if cross != 0:
        return True
    for p in pts[3:]:
        if (p[0] - a[0]) * (c[1] - a[1]) - (p[1] - a[1]) * (c[0] - a[0]) != 0:
            return True
    return False


def _random_initial():
    """Random 12-point set with small integer coordinates."""
    pts = []
    seen = set()
    while len(pts) < 12:
        x = random.randint(-3, 3)
        y = random.randint(-3, 3)
        if (x, y) not in seen:
            seen.add((x, y))
            pts.append([x, y])
    if _not_all_collinear(pts):
        return pts
    return _random_initial()


def _local_search(pts, deadline):
    """Hill-climb / simulated-annealing-style local search."""
    current = [p[:] for p in pts]
    cur_cost = _count_ordinary_lines(current)
    best = [p[:] for p in current]
    best_cost = cur_cost
    start = time.time()
    T = 2.0
    T_min = 0.05
    cooling = 0.995

    while time.time() < deadline:
        if T < T_min:
            T = T_min
        # Random move: pick a point, displace by small integer delta
        idx = random.randrange(12)
        ox, oy = current[idx]
        nx = ox + random.randint(-2, 2)
        ny = oy + random.randint(-2, 2)
        if nx == ox and ny == oy:
            continue
        new_pts = [p[:] for p in current]
        new_pts[idx] = [nx, ny]
        # Check distinctness
        seen = set()
        valid = True
        for p in new_pts:
            t = (p[0], p[1])
            if t in seen:
                valid = False
                break
            seen.add(t)
        if not valid or not _not_all_collinear(new_pts):
            continue
        new_cost = _count_ordinary_lines(new_pts)
        delta_cost = new_cost - cur_cost
        if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
            current = new_pts
            cur_cost = new_cost
            if cur_cost < best_cost:
                best_cost = cur_cost
                best = [p[:] for p in current]
        T *= cooling
    return best, best_cost


def construct():
    deadline = time.time() + _DEADLINE
    random.seed(42)

    # Try several restarts
    best_pts = None
    best_cost = 10**9

    # First, a few deterministic structured starts to seed good regions
    structured_starts = [
        # 3x4 grid (seed)
        [[x, y] for x in range(4) for y in range(3)],
        # 4x3 grid
        [[x, y] for x in range(3) for y in range(4)],
        # A hexagon-like arrangement
        [[0, 0], [1, 0], [2, 0], [3, 0],
         [0, 1], [1, 1], [2, 1], [3, 1],
         [0, 2], [1, 2], [2, 2], [3, 2]],
        # Points on two parallel lines with offset
        [[i, 0] for i in range(6)] + [[i, 1] for i in range(6)],
        # Points on three lines
        [[i, 0] for i in range(4)] + [[i, 1] for i in range(4)] + [[i, 2] for i in range(4)],
    ]

    for s in structured_starts:
        if time.time() > deadline:
            break
        if len(s) != 12:
            continue
        if not _not_all_collinear(s):
            continue
        c = _count_ordinary_lines(s)
        if c < best_cost:
            best_cost = c
            best_pts = [p[:] for p in s]

    # Local search from best and from random starts
    attempts = 0
    while time.time() < deadline and attempts < 50:
        attempts += 1
        if best_pts:
            start = [p[:] for p in best_pts]
        else:
            start = _random_initial()
        res, res_cost = _local_search(start, deadline)
        if res_cost < best_cost:
            best_cost = res_cost
            best_pts = res
        if best_cost <= 6:
            break  # reached theoretical lower bound

    # If we haven't found anything, generate a random valid set
    if best_pts is None:
        best_pts = _random_initial()

    return [[p[0], p[1]] for p in best_pts]
