"""Rewritten: local-search over a 5x5 lattice, counting exact-4-point lines."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

# ---- Fast line-counting kernel -------------------------------------------
# For a fixed 16-point set, a "line" is determined by a pair of points.
# We collect the normalized direction of every pair, then for each direction
# we count how many of the 16 points lie on each of the parallel lines in that
# family, and tally the lines that contain exactly 4 points.

def _norm_dir(dx, dy):
    # canonical direction (dx,dy) -> reduced, sign-canonical
    if dx == 0 and dy == 0:
        return (0, 0)
    g = _gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    return (dx, dy)

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _score(points):
    """Return (num_lines_with_exactly_4, max_collinear) for 16 points."""
    n = len(points)
    # group points by (direction, intercept)
    # intercept for direction (dx,dy): value = dy*x - dx*y  (perpendicular)
    groups = {}
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            d = _norm_dir(dx, dy)
            if d == (0, 0):
                continue
            # perpendicular intercept: dy_norm * x - dx_norm * y
            # using the normalized (dx,dy)
            inter = d[1] * xi - d[0] * yi
            key = (d, inter)
            groups.setdefault(key, 0)
            groups[key] += 1
    # Now, for each direction, we need to know, for each intercept, how many
    # points are on that line. But the pair-count gives C(k,2) for a line with
    # k points. So we recover k from the pair-count.
    # However multiple intercepts share the same direction. We group by direction.
    by_dir = {}
    for (d, inter), c in groups.items():
        by_dir.setdefault(d, []).append((inter, c))
    exact4 = 0
    maxc = 0
    for d, lst in by_dir.items():
        # for each intercept, c = C(k,2) -> k
        for inter, c in lst:
            # solve k*(k-1)/2 = c
            # k = (1 + sqrt(1+8c)) / 2
            s = 1 + 8 * c
            # integer sqrt
            r = int(_isqrt(s))
            if r * r == s and (r + 1) % 2 == 0:
                k = (r + 1) // 2
            else:
                # shouldn't happen for valid pair counts
                continue
            if k > maxc:
                maxc = k
            if k == 4:
                exact4 += 1
    return exact4, maxc

def _isqrt(n):
    if n < 0:
        raise ValueError
    if n == 0:
        return 0
    x = int(n ** 0.5)
    # correct
    while (x + 1) * (x + 1) <= n:
        x += 1
    while x * x > n:
        x -= 1
    return x

# ---- Lattice and helpers --------------------------------------------------
_LATTICE = [(x, y) for x in range(5) for y in range(5)]  # 25 candidate spots
_N = 16

def _initial_random(seed):
    rng = random.Random(seed)
    idx = list(range(25))
    rng.shuffle(idx)
    chosen = idx[:16]
    return [_LATTICE[i] for i in chosen]

def _swap(points, a, b, chosen_set, unused_pool):
    """Swap point a with an unused point from pool. Returns new points list (a copy)."""
    new = list(points)
    new[a] = unused_pool
    return new

# ---- Local search ---------------------------------------------------------
def _local_search(points, rng, deadline):
    # points: list of 16 (x,y)
    # maintain chosen indices for fast unused lookup
    chosen_idx = set(_LATTICE.index(p) for p in points)
    unused = [i for i in range(25) if i not in chosen_idx]
    cur_score, cur_max = _score(points)
    best_score = cur_score
    best_points = list(points)
    if cur_max < 5:
        best_points = list(points)

    # Precompute, for each chosen position a, the list of unused indices
    # (we will iterate over them)
    while time.time() < deadline:
        improved = False
        # random order of positions
        order = list(range(16))
        rng.shuffle(order)
        for a in order:
            if time.time() >= deadline:
                break
            pa = points[a]
            for u in unused:
                pu = _LATTICE[u]
                new_points = list(points)
                new_points[a] = pu
                s, m = _score(new_points)
                if m >= 5:
                    continue
                if s > cur_score:
                    cur_score = s
                    points = new_points
                    chosen_idx.discard(_LATTICE.index(pa))
                    chosen_idx.add(u)
                    unused.remove(u)
                    unused.append(_LATTICE.index(pa))
                    improved = True
                    if s > best_score:
                        best_score = s
                        best_points = list(points)
                    break  # restart position loop after a move
            if improved:
                break
        if not improved:
            # try a few random 2-swaps to escape local optima
            for _ in range(20):
                if time.time() >= deadline:
                    break
                a, b = rng.sample(range(16), 2)
                # swap with two unused? instead do a "kick": replace two chosen with two unused
                u1, u2 = rng.sample(unused, 2)
                new_points = list(points)
                new_points[a] = _LATTICE[u1]
                new_points[b] = _LATTICE[u2]
                s, m = _score(new_points)
                if m >= 5 and s <= cur_score:
                    continue
                if s > cur_score:
                    cur_score = s
                    points = new_points
                    for a2, u2 in [(a, u1), (b, u2)]:
                        old = _LATTICE.index(points[a2])
                        chosen_idx.discard(old)
                        chosen_idx.add(u2)
                        unused.remove(u2)
                        unused.append(old)
                    if s > best_score:
                        best_score = s
                        best_points = list(points)
                    improved = True
                    break
            if not improved:
                # full restart with a fresh random seed-like kick
                # accept a random 3-replacement to jump
                if time.time() >= deadline:
                    break
                ks = rng.sample(range(16), 3)
                us = rng.sample(unused, 3)
                new_points = list(points)
                for k, u in zip(ks, us):
                    new_points[k] = _LATTICE[u]
                s, m = _score(new_points)
                if m < 5:
                    points = new_points
                    for k, u in zip(ks, us):
                        old = _LATTICE.index(points[k])
                        chosen_idx.discard(old)
                        chosen_idx.add(u)
                        unused.remove(u)
                        unused.append(old)
                    cur_score = s
                    if s > best_score:
                        best_score = s
                        best_points = list(points)
                    improved = True
                if not improved:
                    break
    return best_points, best_score

# ---- Main -----------------------------------------------------------------
def construct():
    deadline = time.time() + 140.0  # safety margin under 150s
    best_points = None
    best_score = -1
    seed = 0
    while time.time() < deadline:
        seed += 1
        pts = _initial_random(seed)
        s, m = _score(pts)
        if m < 5:
            pts, s = _local_search(pts, random.Random(seed * 7919 + 13), deadline)
        if s > best_score:
            best_score = s
            best_points = pts
        # if we've hit a strong score, we can stop early to be safe
        if best_score >= 14:
            break
    # Fallback: a known decent grid if nothing found
    if best_points is None:
        best_points = [(x, y) for x in range(4) for y in range(4)]
    # Scale/translate to fit |x|,|y| <= 1000 (already small, but center)
    xs = [p[0] for p in best_points]
    ys = [p[1] for p in best_points]
    mx = max(xs); mn = min(xs)
    my = max(ys); mn2 = min(ys)
    # keep within bounds (they are 0..4, fine)
    return [[p[0], p[1]] for p in best_points]
# EVOLVE-BLOCK-END
