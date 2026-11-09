import time
import random
from itertools import combinations
from fractions import Fraction

# Deadline guard
_T0 = time.time()
_DEADLINE = 150.0
_MARGIN = 10.0  # safety margin

def _time_left():
    return _DEADLINE - _MARGIN - (time.time() - _T0)

def _line_key(p1, p2):
    """Canonical key for the line through p1 and p2 (as a reduced normal)."""
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2:
        return ('v', x1)
    if y1 == y2:
        return ('h', y1)
    # line: a*x + b*y = c, with gcd normalized and sign convention
    a = y2 - y1
    b = x1 - x2
    c = a * x1 + b * y1
    g = _gcd3(a, b, c)
    a //= g; b //= g; c //= g
    # sign convention: make first non-zero of (a,b) positive
    if a < 0 or (a == 0 and b < 0):
        a = -a; b = -b; c = -c
    return ('l', a, b, c)

def _gcd2(a, b):
    a = abs(a); b = abs(b)
    while b:
        a, b = b, a % b
    return a

def _gcd3(a, b, c):
    g = _gcd2(a, b)
    g = _gcd2(g, c)
    return g if g != 0 else 1

def _count_lines(points):
    """Return (num_exactly4, num_five_or_more) for a set of points (list of tuples)."""
    # Map line_key -> set of point indices
    line_map = {}
    pts = list(points)
    n = len(pts)
    for i in range(n):
        for j in range(i+1, n):
            k = _line_key(pts[i], pts[j])
            line_map.setdefault(k, set()).add(i)
    exactly4 = 0
    five_or_more = 0
    for k, idxs in line_map.items():
        if len(idxs) == 4:
            exactly4 += 1
        elif len(idxs) >= 5:
            five_or_more += 1
    return exactly4, five_or_more

def _validate(points):
    """Check constraints: 16 distinct points, |x|,|y|<=1000, no five collinear."""
    if len(points) != 16:
        return False
    seen = set()
    for (x, y) in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    _, five = _count_lines(points)
    if five > 0:
        return False
    return True

def _score(points):
    if not _validate(points):
        return -1
    e4, _ = _count_lines(points)
    return e4

def _random_point(exclude, rng):
    while True:
        x = rng.randint(-1000, 1000)
        y = rng.randint(-1000, 1000)
        if (x, y) not in exclude:
            return (x, y)

def _greedy_build(rng, seed_points=None):
    """Build 16 points greedily, starting from optional seed."""
    if seed_points:
        points = list(seed_points)
    else:
        points = []
    used = set(points)
    while len(points) < 16 and _time_left() > 5.0:
        # Try candidate positions; pick the one that maximizes resulting 4-lines
        # while keeping no-five-collinear.
        best = None
        best_score = -1
        # Sample candidates: some structured, some random
        candidates = []
        # Structured: near existing points
        for (px, py) in points:
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    nx, ny = px + dx, py + dy
                    if abs(nx) <= 1000 and abs(ny) <= 1000 and (nx, ny) not in used:
                        candidates.append((nx, ny))
        # Random
        for _ in range(40):
            candidates.append(_random_point(used, rng))
        # Deduplicate
        candidates = list(set(candidates))
        for cand in candidates:
            trial = points + [cand]
            _, five = _count_lines(trial)
            if five > 0:
                continue
            e4, _ = _count_lines(trial)
            if e4 > best_score:
                best_score = e4
                best = cand
        if best is None:
            # Fallback: any valid point
            for _ in range(200):
                cand = _random_point(used, rng)
                trial = points + [cand]
                _, five = _count_lines(trial)
                if five == 0:
                    best = cand
                    break
            if best is None:
                # Give up building, pad with random valid if possible
                break
        points.append(best)
        used.add(best)
    # Pad if needed
    while len(points) < 16:
        cand = _random_point(used, rng)
        trial = points + [cand]
        _, five = _count_lines(trial)
        if five == 0:
            points.append(cand)
            used.add(cand)
        else:
            # try again
            continue
    return points

def _local_search(points, rng, max_iters=2000):
    """Move individual points to improve the 4-line count."""
    pts = [tuple(p) for p in points]
    best_score = _score(pts)
    if best_score < 0:
        return pts
    n = len(pts)
    iters = 0
    while iters < max_iters and _time_left() > 5.0:
        iters += 1
        i = rng.randrange(n)
        old = pts[i]
        # Candidate new positions
        cands = []
        # Nearby moves
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = old[0] + dx, old[1] + dy
                if abs(nx) <= 1000 and abs(ny) <= 1000:
                    cands.append((nx, ny))
        # Some random
        for _ in range(10):
            cands.append(_random_point(set(pts) - {old}, rng))
        cands = list(set(cands))
        improved = False
        for cand in cands:
            if cand == old:
                continue
            if cand in set(pts) - {old}:
                continue
            trial = list(pts)
            trial[i] = cand
            s = _score(trial)
            if s > best_score:
                best_score = s
                pts = trial
                improved = True
                break
        if not improved:
            # occasional random restart of one point to escape
            pass
    return pts

def _grid_seed():
    """4x4 grid as a starting structure."""
    return [(x, y) for x in range(4) for y in range(4)]

def construct():
    global _T0
    _T0 = time.time()
    rng = random.Random(12345)
    best_points = None
    best_score = -1

    # Start from the 4x4 grid and improve
    grid = _grid_seed()
    s = _score(grid)
    if s > best_score:
        best_score = s
        best_points = list(grid)

    # Local search from grid
    improved = _local_search(grid, rng, max_iters=3000)
    s = _score(improved)
    if s > best_score:
        best_score = s
        best_points = list(improved)

    # Try greedy builds from scratch with different seeds
    for trial in range(60):
        if _time_left() < 20.0:
            break
        rng2 = random.Random(1000 + trial)
        built = _greedy_build(rng2)
        s = _score(built)
        if s > best_score:
            best_score = s
            best_points = list(built)
            # polish
            improved = _local_search(built, rng2, max_iters=2000)
            s2 = _score(improved)
            if s2 > best_score:
                best_score = s2
                best_points = list(improved)

    # Final polish on best
    if _time_left() > 15.0:
        improved = _local_search(best_points, rng, max_iters=4000)
        s = _score(improved)
        if s > best_score:
            best_score = s
            best_points = list(improved)

    if best_points is None:
        # Fallback: return grid
        best_points = list(grid)

    # Ensure valid
    if not _validate(best_points):
        best_points = list(grid)

    return [[x, y] for (x, y) in best_points]
