import time
import math
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def _clamp(v):
    return -1000 if v < -1000 else (1000 if v > 1000 else v)

def _line_key(xi, yi, xj, yj):
    dx = xj - xi
    dy = yj - yi
    g = math.gcd(abs(dx), abs(dy))
    if g == 0:
        return None
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    A = dy
    B = -dx
    C = dy * xi - dx * yi
    gg = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if gg:
        A //= gg
        B //= gg
        C //= gg
    if A < 0 or (A == 0 and B < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _lines_dict_fast(pts):
    lines = defaultdict(set)
    n = len(pts)
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            key = _line_key(xi, yi, xj, yj)
            if key is not None:
                lines[key].add(i)
                lines[key].add(j)
    return lines

def _valid(pts):
    if len(pts) != 16:
        return False
    if len(set(tuple(p) for p in pts)) != 16:
        return False
    for x, y in pts:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
    return True

def _obj(pts):
    if not _valid(pts):
        return -10 ** 9
    lines = _lines_dict_fast(pts)
    c4 = 0
    for s in lines.values():
        c = len(s)
        if c == 4:
            c4 += 1
        elif c >= 5:
            return -10 ** 9
    return c4

def _free_pos_on_line(pts, idxs, occupied):
    if len(idxs) < 2:
        return None
    a = pts[idxs[0]]
    b = pts[idxs[1]]
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    g = math.gcd(abs(dx), abs(dy))
    if g == 0:
        return None
    dx //= g
    dy //= g
    bx, by = a
    for t in range(-80, 81):
        x = bx + t * dx
        y = by + t * dy
        if abs(x) > 1000 or abs(y) > 1000:
            continue
        if (x, y) in occupied:
            continue
        return [x, y]
    return None

def _small_perturb_at(pts, rng, idx, radius=5):
    dx = rng.randint(-radius, radius)
    dy = rng.randint(-radius, radius)
    if dx == 0 and dy == 0:
        dx = rng.choice([-1, 1])
    cand = pts[:]
    cand[idx] = [_clamp(cand[idx][0] + dx), _clamp(cand[idx][1] + dy)]
    return cand

def _make_move(pts, rng):
    r = rng.random()
    if r < 0.45:
        return _small_perturb_at(pts, rng, rng.randrange(16), radius=4)
    elif r < 0.65:
        lines = _lines_dict_fast(pts)
        three = [s for s in lines.values() if len(s) == 3]
        if three:
            s = rng.choice(three)
            others = [i for i in range(16) if i not in s]
            idx = rng.choice(others)
            occupied = set(tuple(p) for p in pts)
            pos = _free_pos_on_line(pts, list(s), occupied)
            if pos is not None:
                cand = pts[:]
                cand[idx] = pos
                return cand
        return _small_perturb_at(pts, rng, rng.randrange(16), radius=4)
    elif r < 0.80:
        lines = _lines_dict_fast(pts)
        five = [s for s in lines.values() if len(s) >= 5]
        if five:
            s = rng.choice(five)
            idx = rng.choice(list(s))
            return _small_perturb_at(pts, rng, idx, radius=6)
        return _small_perturb_at(pts, rng, rng.randrange(16), radius=4)
    elif r < 0.90:
        lines = _lines_dict_fast(pts)
        four = [s for s in lines.values() if len(s) == 4]
        if four:
            s = rng.choice(four)
            others = [i for i in range(16) if i not in s]
            idx = rng.choice(others)
            occupied = set(tuple(p) for p in pts)
            pos = _free_pos_on_line(pts, list(s), occupied)
            if pos is not None:
                cand = pts[:]
                cand[idx] = pos
                return cand
        return _small_perturb_at(pts, rng, rng.randrange(16), radius=4)
    else:
        idx = rng.randrange(16)
        cand = pts[:]
        lim = rng.choice([30, 60, 100])
        cand[idx] = [rng.randint(-lim, lim), rng.randint(-lim, lim)]
        return cand

def _jump(pts, rng):
    cand = pts[:]
    k = rng.randint(3, 6)
    idxs = rng.sample(range(16), k)
    for idx in idxs:
        lim = rng.choice([20, 40, 60])
        cand[idx] = [rng.randint(-lim, lim), rng.randint(-lim, lim)]
    return cand

def _ils(start, end, rng, T0=3.0):
    cur = [list(p) for p in start]
    cur_s = _obj(cur)
    best = [list(p) for p in cur]
    best_s = cur_s
    T = T0
    stuck = 0
    min_T = 0.01
    while time.time() < end:
        cand = _make_move(cur, rng)
        ns = _obj(cand)
        if ns >= cur_s or (T > 0.0 and rng.random() < math.exp((ns - cur_s) / T)):
            cur = cand
            cur_s = ns
            if cur_s > best_s:
                best = [list(p) for p in cur]
                best_s = cur_s
                stuck = 0
            else:
                stuck += 1
        else:
            stuck += 1
        T = max(min_T, T * 0.9999)
        if stuck > 800:
            cur = _jump(cur, rng)
            cur_s = _obj(cur)
            T = max(1.5, T0 * 0.6)
            stuck = 0
    return best, best_s

def _greedy_polish(start, end, rng):
    cur = [list(p) for p in start]
    cur_s = _obj(cur)
    best = [list(p) for p in cur]
    best_s = cur_s
    while time.time() < end:
        cand = _make_move(cur, rng)
        ns = _obj(cand)
        if ns > cur_s:
            cur = cand
            cur_s = ns
            if cur_s > best_s:
                best = [list(p) for p in cur]
                best_s = cur_s
    return best, best_s

def _seed_grid4():
    return [[x, y] for x in range(4) for y in range(4)]

def _seed_subset(n, k, rng):
    m = n // 2
    pts = [[x, y] for x in range(-m, m + 1) for y in range(-m, m + 1)]
    rng.shuffle(pts)
    return [list(p) for p in pts[:k]]

def _seed_random(k, rng, lim):
    seen = set()
    pts = []
    tries = 0
    while len(pts) < k and tries < 10000:
        tries += 1
        p = (rng.randint(-lim, lim), rng.randint(-lim, lim))
        if p not in seen:
            seen.add(p)
            pts.append(list(p))
    while len(pts) < k:
        pts.append([rng.randint(-lim, lim), rng.randint(-lim, lim)])
    return pts

def _seed_known_good(rng):
    configs = [
        [[x, y] for x in range(4) for y in range(4)],
        [[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[4,1],[0,2],[1,2],[3,2],[4,2],[0,3],[2,3],[3,3],[5,3]],
        [[0,0],[1,0],[2,0],[3,0],[0,2],[1,2],[2,2],[4,2],[0,4],[1,4],[3,4],[4,4],[0,6],[2,6],[3,6],[5,6]],
        [[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1],[0,2],[1,2],[3,2],[4,2],[0,3],[2,3],[3,3],[4,3]],
    ]
    cfg = rng.choice(configs)
    pts = [list(p) for p in cfg]
    for _ in range(rng.randint(0, 3)):
        idx = rng.randrange(16)
        pts[idx][0] += rng.randint(-2, 2)
        pts[idx][1] += rng.randint(-2, 2)
        pts[idx][0] = _clamp(pts[idx][0])
        pts[idx][1] = _clamp(pts[idx][1])
    seen = set()
    unique = []
    for p in pts:
        if tuple(p) not in seen:
            seen.add(tuple(p))
            unique.append(p)
    while len(unique) < 16:
        p = [rng.randint(-5, 5), rng.randint(-5, 5)]
        if tuple(p) not in seen:
            seen.add(tuple(p))
            unique.append(p)
    return unique

def _construct():
    t0 = time.time()
    B = 120.0
    rng = random.Random(20240607)

    seeds = [
        _seed_grid4(),
        _seed_known_good(rng),
        _seed_subset(7, 16, rng),
        _seed_subset(8, 16, rng),
        _seed_subset(6, 16, rng),
        _seed_random(16, rng, 15),
        _seed_random(16, rng, 30),
    ]

    quick_budget = 0.20 * B
    n = len(seeds)
    best = None
    best_s = -1
    for i, s in enumerate(seeds):
        end_i = t0 + (i + 1) * quick_budget / n
        b, bs = _ils(s, end_i, rng, T0=2.5)
        if bs > best_s:
            best_s = bs
            best = b

    if best is None:
        best = _seed_grid4()
        best_s = _obj(best)

    long_end = t0 + 0.75 * B
    best, best_s = _ils(best, long_end, rng, T0=3.5)

    restart_end = t0 + 0.92 * B
    while time.time() < restart_end:
        jumped = _jump(best, rng)
        b, bs = _ils(jumped, min(time.time() + 5.0, restart_end), rng, T0=2.0)
        if bs > best_s:
            best_s = bs
            best = b

    polish_end = t0 + 0.98 * B
    best, best_s = _greedy_polish(best, polish_end, rng)

    if not _valid(best) or best_s < 0:
        fallback = _seed_grid4()
        return fallback

    return [[int(x), int(y)] for x, y in best]

# EVOLVE-BLOCK-END

def construct():
    return _construct()
