"""Rewritten program: seed a strong 4x4-ish configuration, then time-boxed local search."""
import itertools
import time
import random

_DEADLINE_FRAC = 0.80  # safety margin vs 150s wall-clock


def _time_left(start, frac=_DEADLINE_FRAC):
    # assume 150s budget; return seconds left with safety margin
    return (start + 150.0 * frac) - time.time()


def _line_key(p1, p2):
    """Canonical (dx, dy) direction for the line through two integer points, reduced by gcd."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return (0, 0)
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    g = _gcd(abs(dx), abs(dy))
    return (dx // g, dy // g)


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _on_line(p, a, b):
    """Check if p is collinear with a and b (integer cross product)."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) == 0


def _evaluate(points):
    """Return (num_exact4_lines, num_lines_with_5plus). Lower 5plus is better; we forbid >0."""
    n = len(points)
    exact4 = 0
    five_plus = 0
    # For each pair, determine the line; count points on it.
    # To avoid double counting lines, canonicalize by (key, min-point-on-line).
    seen = set()
    for i in range(n):
        for j in range(i + 1, n):
            a, b = points[i], points[j]
            key = _line_key(a, b)
            if key == (0, 0):
                continue
            # canonical representative: the lexicographically smallest point on this line
            pts_on = []
            for k in range(n):
                p = points[k]
                if _on_line(p, a, b):
                    pts_on.append(p)
            if len(pts_on) < 4:
                continue
            # canonical id
            rep = min(pts_on)
            line_id = (key, rep)
            if line_id in seen:
                continue
            seen.add(line_id)
            if len(pts_on) == 4:
                exact4 += 1
            else:
                five_plus += 1
    return exact4, five_plus


def _valid(points):
    exact4, five_plus = _evaluate(points)
    return five_plus == 0, exact4


def _seed_points():
    # A 4x4 grid gives 4 rows + 4 cols + 2 diagonals = 10 exact-4 lines, no 5 collinear.
    pts = [[x, y] for x in range(4) for y in range(4)]
    return pts


def _neighbor_candidates(current, rng, limit=20000):
    """Yield candidate replacement points to try swapping with one of the current points."""
    # Random integer points in a bounded box around current extent
    xs = [p[0] for p in current]
    ys = [p[1] for p in current]
    xlo, xhi = min(xs) - 3, max(xs) + 3
    ylo, yhi = min(ys) - 3, max(ys) + 3
    # Also allow a wider random box occasionally
    for _ in range(limit):
        if rng.random() < 0.7:
            x = rng.randint(xlo, xhi)
            y = rng.randint(ylo, yhi)
        else:
            x = rng.randint(-12, 12)
            y = rng.randint(-12, 12)
        yield (x, y)


def _local_search(points, start, rng, max_iters=None):
    best = list(points)
    _, best_score = _evaluate(best)
    it = 0
    while _time_left(start) > 1.0:
        if max_iters is not None and it >= max_iters:
            break
        it += 1
        improved = False
        # Try swapping each point with a few random candidates
        for idx in range(len(best)):
            if _time_left(start) < 1.0:
                break
            orig = best[idx]
            # Try a handful of candidate replacements for this point
            for cand in _neighbor_candidates(best, rng, limit=40):
                if cand == orig:
                    continue
                trial = best[:]
                trial[idx] = list(cand)
                # quick distinctness check
                if len(set(map(tuple, trial))) != len(trial):
                    continue
                ok, score = _valid(trial)
                if ok and score > best_score:
                    best = trial
                    best_score = score
                    improved = True
                    break
            if improved:
                break
        if not improved:
            # occasional random restart around best to escape local optima
            if _time_left(start) > 5.0 and rng.random() < 0.3:
                # perturb a couple of points
                trial = best[:]
                for _ in range(2):
                    i = rng.randrange(len(trial))
                    x = trial[i][0] + rng.randint(-4, 4)
                    y = trial[i][1] + rng.randint(-4, 4)
                    trial[i] = [x, y]
                ok, score = _valid(trial)
                if ok and score >= best_score:
                    best = trial
                    best_score = score
    return best, best_score


def construct():
    start = time.time()
    rng = random.Random(12345)
    best = _seed_points()
    _, best_score = _evaluate(best)

    # Local search
    best, best_score = _local_search(best, start, rng)

    # Final safety: ensure validity; if not, fall back to seed
    ok, score = _valid(best)
    if not ok:
        best = _seed_points()
        score = 10

    return [list(p) for p in best]
