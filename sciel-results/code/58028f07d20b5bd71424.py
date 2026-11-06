import time
import random
import sys
from math import gcd

DEADLINE = 145  # safety margin before 150s kill

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    a = -dy
    b = dx
    c = dx * y1 - dy * x1
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
        c = -c
    g2 = gcd(abs(a), abs(b))
    if g2 > 1:
        a //= g2
        b //= g2
        c //= g2
    return (a, b, c)

def _build_lines(points):
    n = len(points)
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            s = line_map.get(key)
            if s is None:
                s = set()
                line_map[key] = s
            s.add(i)
            s.add(j)
    lines = []
    line_masks = []
    for s in line_map.values():
        if len(s) >= 3:
            lines.append(frozenset(s))
            m = 0
            for idx in s:
                m |= 1 << idx
            line_masks.append(m)
    return lines, len(lines), line_masks

def _check_no_four(points, lines):
    for L in lines:
        if len(L) >= 4:
            return False
    return True

def _max_cap_free(points, lines, line_masks, line_count, time_limit=5.0):
    """Exact max 3-collinear-free subset via branch and bound.
    
    We want the largest subset S such that no three points in S are collinear.
    Equivalently, we assign each point color 1 (in S) or 0 (not in S) such that
    no line has 3 or more points all colored 1.
    
    We use a recursive search: process points in order, for each point decide
    in/out. A line becomes "saturated" (violating) if it gets 3 points colored 1.
    """
    n = len(points)
    start = time.time()
    timed_out = [False]

    # For each point, the lines it belongs to (by line index)
    point_lines = [[] for _ in range(n)]
    for li, L in enumerate(lines):
        for idx in L:
            point_lines[idx].append(li)

    # Line info: how many points are colored 1 on each line
    line_size = [len(L) for L in lines]
    line_ones = [0] * line_count  # count of points colored 1 on each line

    color = [0] * n
    best = [0]
    cur = [0]

    # Precompute: for each line, the sorted list of points on it
    line_points = [sorted(L) for L in lines]

    def bound(idx):
        # Upper bound: current count + remaining points
        # Tighter: for each line, if it already has k ones and total size m,
        # we can add at most (m - k) more to this line without exceeding... 
        # Actually the constraint is: no line can have 3+ ones.
        # So for each line with k ones, we can add at most (2 - k) more ones to it.
        # The remaining points can be in S only if they don't violate any line.
        # Simple bound: cur + (n - idx)
        return cur[0] + (n - idx)

    def dfs(idx):
        if timed_out[0]:
            return
        if time.time() - start > time_limit:
            timed_out[0] = True
            return
        if idx == n:
            if cur[0] > best[0]:
                best[0] = cur[0]
            return
        if bound(idx) <= best[0]:
            return
        p = idx
        # Try in (color 1)
        ok = True
        touched = []
        for li in point_lines[p]:
            if line_ones[li] >= 2:
                ok = False
                break
            line_ones[li] += 1
            touched.append(li)
        if ok:
            color[p] = 1
            cur[0] += 1
            dfs(idx + 1)
            cur[0] -= 1
            for li in touched:
                line_ones[li] -= 1
            color[p] = 0
        if timed_out[0]:
            return
        # Try out (color 0)
        color[p] = 0
        dfs(idx + 1)

    dfs(0)
    return best[0]

def _score(points, time_limit=5.0):
    lines, lc, lm = _build_lines(points)
    if not _check_no_four(points, lines):
        return 999
    return _max_cap_free(points, lines, lm, lc, time_limit)

def _valid(points):
    if len(set(tuple(p) for p in points)) != 20:
        return False
    if any(abs(p[0]) > 100 or abs(p[1]) > 100 for p in points):
        return False
    lines, lc, lm = _build_lines(points)
    return _check_no_four(points, lines)

def _generate_parabola_perturbed(seed):
    rng = random.Random(seed)
    pts = []
    for i in range(20):
        x = -10 + i
        y = x * x // 4 + rng.randint(-3, 3)
        pts.append([x, y])
    return pts

def _generate_random(seed):
    rng = random.Random(seed)
    pts = []
    for i in range(20):
        pts.append([rng.randint(-100, 100), rng.randint(-100, 100)])
    return pts

def _generate_lattice(seed):
    """Generate points on a small lattice with perturbations."""
    rng = random.Random(seed)
    pts = []
    # Use a grid of size ~5x5 with some jitter
    for i in range(20):
        gx = (i % 5) * 20 - 40 + rng.randint(-3, 3)
        gy = (i // 5) * 20 - 40 + rng.randint(-3, 3)
        pts.append([gx, gy])
    return pts

def _local_search(points, time_budget, rng=None):
    """Hill climbing: move one point at a time to reduce score."""
    if rng is None:
        rng = random.Random(42)
    start = time.time()
    cur = [p[:] for p in points]
    lines, lc, lm = _build_lines(cur)
    if not _check_no_four(cur, lines):
        return cur, 999
    cur_score = _max_cap_free(cur, lines, lm, lc, 3.0)
    best = [p[:] for p in cur]
    best_score = cur_score
    no_improve = 0
    
    while time.time() - start < time_budget:
        improved = False
        # Try moving each point to a random nearby location
        for _ in range(50):
            i = rng.randrange(20)
            ox, oy = cur[i][:]
            cur[i] = [rng.randint(ox - 5, ox + 5), rng.randint(oy - 5, oy + 5)]
            if abs(cur[i][0]) > 100 or abs(cur[i][1]) > 100:
                cur[i] = [ox, oy]
                continue
            if len(set(tuple(p) for p in cur)) != 20:
                cur[i] = [ox, oy]
                continue
            lines, lc, lm = _build_lines(cur)
            if not _check_no_four(cur, lines):
                cur[i] = [ox, oy]
                continue
            s = _max_cap_free(cur, lines, lm, lc, 2.0)
            if s < best_score:
                best_score = s
                best = [p[:] for p in cur]
                improved = True
                no_improve = 0
            else:
                cur[i] = [ox, oy]
        if not improved:
            no_improve += 1
            if no_improve > 10:
                break
    return best, best_score

def _two_point_search(points, time_budget, rng=None):
    """Try swapping pairs of points."""
    if rng is None:
        rng = random.Random(99)
    start = time.time()
    cur = [p[:] for p in points]
    lines, lc, lm = _build_lines(cur)
    if not _check_no_four(cur, lines):
        return cur, 999
    cur_score = _max_cap_free(cur, lines, lm, lc, 3.0)
    best = [p[:] for p in cur]
    best_score = cur_score
    
    while time.time() - start < time_budget:
        improved = False
        a, b = rng.sample(range(20), 2)
        for _ in range(8):
            na = [rng.randint(cur[a][0] - 6, cur[a][0] + 6), rng.randint(cur[a][1] - 6, cur[a][1] + 6)]
            nb = [rng.randint(cur[b][0] - 6, cur[b][0] + 6), rng.randint(cur[b][1] - 6, cur[b][1] + 6)]
            if abs(na[0]) > 100 or abs(na[1]) > 100 or abs(nb[0]) > 100 or abs(nb[1]) > 100:
                continue
            ox, oy = cur[a][:]
            px, py = cur[b][:]
            cur[a] = na
            cur[b] = nb
            if len(set(tuple(p) for p in cur)) != 20:
                cur[a] = [ox, oy]
                cur[b] = [px, py]
                continue
            lines, lc, lm = _build_lines(cur)
            if not _check_no_four(cur, lines):
                cur[a] = [ox, oy]
                cur[b] = [px, py]
                continue
            s = _max_cap_free(cur, lines, lm, lc, 2.0)
            if s < best_score:
                best_score = s
                best = [p[:] for p in cur]
                improved = True
            else:
                cur[a] = [ox, oy]
                cur[b] = [px, py]
        if not improved and rng.random() < 0.3:
            continue
    return best, best_score

def construct():
    start_time = time.time()
    best_points = None
    best_score = 999

    # Try multiple initialization strategies
    generators = [
        (_generate_parabola_perturbed, 60),
        (_generate_random, 60),
        (_generate_lattice, 40),
    ]

    for gen_func, num_seeds in generators:
        for seed in range(num_seeds):
            if time.time() - start_time > DEADLINE - 20:
                break
            pts = gen_func(seed)
            if not _valid(pts):
                continue
            lines, lc, lm = _build_lines(pts)
            s = _max_cap_free(pts, lines, lm, lc, 3.0)
            if s < best_score:
                best_score = s
                best_points = [p[:] for p in pts]
                if s <= 17 and time.time() - start_time < DEADLINE - 30:
                    # Refine with local search
                    remaining = DEADLINE - 35 - (time.time() - start_time)
                    if remaining > 5:
                        refined, rs = _local_search(pts, min(15.0, remaining))
                        if rs < best_score:
                            best_score = rs
                            best_points = refined
                        if best_score <= 16 and time.time() - start_time < DEADLINE - 15:
                            remaining2 = DEADLINE - 15 - (time.time() - start_time)
                            if remaining2 > 5:
                                refined2, rs2 = _two_point_search(best_points, min(10.0, remaining2))
                                if rs2 < best_score:
                                    best_score = rs2
                                    best_points = refined2

    if best_points is None or best_score >= 20:
        # Fallback: parabola
        best_points = [[x, x * x // 4] for x in range(-10, 10)]
    
    return best_points
