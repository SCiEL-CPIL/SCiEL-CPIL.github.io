"""Rewritten program: local search to maximize max multiplicity of positive doubled area."""
import random
import itertools
import time

_START = time.time()
_DEADLINE = 55.0  # safety margin before 60s kill (evaluator kills at 90s, keep margin)


def _doubled_areas(points):
    """Return dict: doubled_area -> count, for all non-degenerate triples."""
    n = len(points)
    counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            for k in range(j + 1, n):
                x3, y3 = points[k]
                d = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
                ad = abs(d)
                if ad > 0:
                    counts[ad] = counts.get(ad, 0) + 1
    return counts


def _score(points):
    """Return the largest multiplicity of a single positive doubled area."""
    counts = _doubled_areas(points)
    if not counts:
        return 0
    return max(counts.values())


def _free_locations(used):
    """Yield candidate (x,y) not in used, within [-30,30]."""
    used_set = set(used)
    for x in range(-30, 31):
        for y in range(-30, 31):
            if (x, y) not in used_set:
                yield (x, y)


def _local_search(points, rng, budget):
    """Hill-climb by swapping a single point to a new location. Returns best points."""
    pts = list(points)
    n = len(pts)
    best_score = _score(pts)
    best_pts = list(pts)
    used = set(pts)
    # Precompute candidate locations once (bounded grid)
    all_locs = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    while time.time() - _START < budget:
        # pick a random index to move
        i = rng.randrange(n)
        old = pts[i]
        # pick a random new location
        new = rng.choice(all_locs)
        if new in used:
            continue
        pts[i] = new
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = list(pts)
            used.discard(old)
            used.add(new)
        else:
            # revert
            pts[i] = old
    return best_pts


def _seed_config(rng):
    """Generate a random seed configuration of 30 distinct points."""
    pts = []
    used = set()
    while len(pts) < 30:
        x = rng.randint(-30, 30)
        y = rng.randint(-30, 30)
        if (x, y) not in used:
            used.add((x, y))
            pts.append((x, y))
    return pts


def _structured_seed():
    """A structured seed: points on two parallel lines to encourage equal areas."""
    pts = []
    # 15 points on y=0, 15 on y=1
    for x in range(0, 30, 2):
        pts.append((x, 0))
    for x in range(0, 30, 2):
        pts.append((x, 1))
    return pts


def construct():
    rng = random.Random(12345)
    budget = _DEADLINE
    # Start with a structured seed that already has high multiplicity
    best_pts = _structured_seed()
    best_score = _score(best_pts)

    # Also try the grid seed
    grid_pts = [(x, y) for x in range(6) for y in range(5)]
    grid_score = _score(grid_pts)
    if grid_score > best_score:
        best_score = grid_score
        best_pts = list(grid_pts)

    # Local search from the best seed
    remaining = budget - (time.time() - _START)
    if remaining > 5:
        improved = _local_search(best_pts, rng, time.time() + remaining - 5)
        s = _score(improved)
        if s > best_score:
            best_score = s
            best_pts = improved

    # A few random restarts if time remains
    while time.time() - _START < budget - 8:
        seed = _seed_config(rng)
        s = _score(seed)
        if s > best_score:
            best_score = s
            best_pts = list(seed)
            # quick local search
            improved = _local_search(best_pts, rng, time.time() + 4)
            s2 = _score(improved)
            if s2 > best_score:
                best_score = s2
                best_pts = improved
        if time.time() - _START > budget - 10:
            break

    # Ensure exactly 30 distinct integer pairs in range
    assert len(best_pts) == 30
    assert len(set(best_pts)) == 30
    for (x, y) in best_pts:
        assert -30 <= x <= 30 and -30 <= y <= 30
    return best_pts
