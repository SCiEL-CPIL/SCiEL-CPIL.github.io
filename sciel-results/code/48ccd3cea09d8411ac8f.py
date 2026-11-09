"""New design: structured seeds + fast local search with deadline guard."""
import time
import random
from math import gcd
from collections import defaultdict

DEADLINE = 135.0  # safety margin under 150s kill

def _line_key(p1, p2):
    """Canonical key for the line through two integer points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    g = gcd(abs(dx), abs(dy))
    if g:
        dx //= g
        dy //= g
    # Canonical direction: make dx>0 or (dx==0 and dy>0)
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # Normal form: line is (dy)*x - (dx)*y = c
    c = dy * x1 - dx * y1
    # Canonical sign for (dy, -dx): make first nonzero positive
    a, b = dy, -dx
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
        c = -c
    return (a, b, c)

def _count_exact4(pts):
    """Count lines containing exactly 4 of the given points. Returns (count, has_five)."""
    n = len(pts)
    line_pts = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(pts[i], pts[j])
            line_pts[key].add(i)
    count = 0
    has_five = False
    for key, idxs in line_pts.items():
        s = len(idxs)
        if s == 4:
            count += 1
        elif s >= 5:
            has_five = True
    return count, has_five

def _score(pts):
    """Objective: number of exact-4 lines, minus huge penalty if any 5+ collinear."""
    count, has_five = _count_exact4(pts)
    if has_five:
        return count - 10000
    return count

def _seed_grid44():
    return [[x, y] for x in range(4) for y in range(4)]

def _seed_grid55_subset():
    # Take a 5x5 grid and pick 16 points; try several patterns
    base = [[x, y] for x in range(5) for y in range(5)]
    # pattern: all of rows 0-3 except a few, plus extras
    return base[:16]

def _random_seed(rng, size=1000):
    pts = []
    while len(pts) < 16:
        p = [rng.randint(-size, size), rng.randint(-size, size)]
        if p not in pts:
            pts.append(p)
    return pts

def _local_search(pts, rng, deadline, initial_score):
    """Simulated-annealing-style local search. Returns best pts and score."""
    best = [list(p) for p in pts]
    best_score = initial_score
    cur = [list(p) for p in pts]
    cur_score = initial_score
    temp = 3.0
    cooling = 0.995
    n = len(cur)
    max_iter = 400000
    it = 0
    while it < max_iter:
        if time.time() > deadline:
            break
        it += 1
        # Pick a random point and a random move
        idx = rng.randrange(n)
        # Choose move type
        r = rng.random()
        if r < 0.5:
            # jitter: small random offset
            ox, oy = cur[idx]
            nx = ox + rng.randint(-3, 3)
            ny = oy + rng.randint(-3, 3)
            if abs(nx) > 1000 or abs(ny) > 1000:
                continue
            new_pt = [nx, ny]
        else:
            # reposition to a random nearby grid point
            ox, oy = cur[idx]
            nx = ox + rng.randint(-6, 6)
            ny = oy + rng.randint(-6, 6)
            if abs(nx) > 1000 or abs(ny) > 1000:
                continue
            new_pt = [nx, ny]
        # Skip if duplicate
        if new_pt in cur:
            continue
        old = cur[idx]
        cur[idx] = new_pt
        new_score = _score(cur)
        delta = new_score - cur_score
        if delta >= 0 or rng.random() < (temp / max(1.0, abs(delta)) if delta < 0 else 1.0):
            cur_score = new_score
            if cur_score > best_score:
                best_score = cur_score
                best = [list(p) for p in cur]
        else:
            cur[idx] = old
        temp *= cooling
    return best, best_score

def construct():
    start = time.time()
    deadline = start + DEADLINE
    rng = random.Random(12345)

    seeds = []
    # 4x4 grid
    s = _seed_grid44()
    seeds.append(s)
    # 4x4 grid shifted
    s2 = [[x+2, y+2] for x in range(4) for y in range(4)]
    seeds.append(s2)
    # 5x5 grid first 16
    s3 = [[x, y] for x in range(5) for y in range(5)][:16]
    seeds.append(s3)
    # 5x5 grid rows 0-2 full (15) + one from row 3
    s4 = [[x, y] for x in range(5) for y in range(3)]
    s4.append([0, 3])
    seeds.append(s4)
    # A few random seeds
    for _ in range(4):
        seeds.append(_random_seed(rng, size=30))

    best_pts = None
    best_score = -1
    for seed in seeds:
        if time.time() > deadline:
            break
        sc = _score(seed)
        if sc > best_score:
            best_score = sc
            best_pts = [list(p) for p in seed]
        # local search from this seed
        bp, bs = _local_search(seed, rng, deadline, sc)
        if bs > best_score:
            best_score = bs
            best_pts = bp

    # If we have spare time, do a few more random-restart local searches
    if best_pts is None:
        best_pts = _seed_grid44()
        best_score = _score(best_pts)

    # Final safety: ensure validity (no five collinear, distinct, bounds)
    # If invalid, fall back to 4x4 grid
    count, has_five = _count_exact4(best_pts)
    if has_five or len(set(tuple(p) for p in best_pts)) != 16:
        best_pts = _seed_grid44()

    return best_pts
