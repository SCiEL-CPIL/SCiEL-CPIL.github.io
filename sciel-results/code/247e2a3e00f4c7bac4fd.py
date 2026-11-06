import time
import random
import math
from collections import defaultdict

# EVOLVE-BLOCK-START

def _score(points):
    n = len(points)
    if n != 16:
        return -1
    if len(set(tuple(p) for p in points)) != 16:
        return -1
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return -1
    line_counts = defaultdict(int)
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            a = y2 - y1
            b = x1 - x2
            c = a * x1 + b * y1
            g = math.gcd(math.gcd(abs(a), abs(b)), abs(c)) if (a or b or c) else 1
            if g:
                a //= g
                b //= g
                c //= g
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
                c = -c
            line_counts[(a, b, c)] += 1
    exact4 = 0
    for pc in line_counts.values():
        disc = 1 + 8 * pc
        s = math.isqrt(disc)
        if s * s != disc:
            continue
        k = (1 + s) // 2
        if k * (k - 1) // 2 != pc:
            continue
        if k >= 5:
            return -k
        if k == 4:
            exact4 += 1
    return exact4


def _mutate(points, rng, radius):
    new_points = [p[:] for p in points]
    idx = rng.randrange(16)
    cx, cy = new_points[idx]
    for _ in range(30):
        x = cx + rng.randint(-radius, radius)
        y = cy + rng.randint(-radius, radius)
        x = max(-1000, min(1000, x))
        y = max(-1000, min(1000, y))
        cand = (x, y)
        ok = True
        for j, p in enumerate(new_points):
            if j != idx and p[0] == x and p[1] == y:
                ok = False
                break
        if ok:
            new_points[idx] = [x, y]
            return new_points
    return new_points


def _swap(points, rng):
    new_points = [p[:] for p in points]
    i, j = rng.sample(range(16), 2)
    new_points[i], new_points[j] = new_points[j], new_points[i]
    return new_points


def _initial_grid():
    return [[x, y] for x in range(4) for y in range(4)]


def _initial_random(rng, spread=30):
    pts = []
    used = set()
    while len(pts) < 16:
        p = [rng.randint(-spread, spread), rng.randint(-spread, spread)]
        if (p[0], p[1]) not in used:
            used.add((p[0], p[1]))
            pts.append(p)
    return pts


def construct():
    deadline = time.time() + 140.0
    rng = random.Random(42)

    best_points = _initial_grid()
    best_score = _score(best_points)

    inits = [
        _initial_grid(),
        _initial_random(rng, 20),
        _initial_random(rng, 50),
        _initial_random(rng, 100),
    ]

    for init in inits:
        if time.time() >= deadline:
            break
        cur = [p[:] for p in init]
        cur_score = _score(cur)
        if cur_score > best_score:
            best_score = cur_score
            best_points = [p[:] for p in cur]

        T = 2.0
        cooling = 0.999
        min_T = 0.05
        radius = 8

        while time.time() < deadline and T > min_T:
            for _ in range(150):
                if time.time() >= deadline:
                    break
                if rng.random() < 0.25:
                    cand = _swap(cur, rng)
                else:
                    cand = _mutate(cur, rng, radius)
                cs = _score(cand)
                if cs > cur_score:
                    cur = cand
                    cur_score = cs
                    if cs > best_score:
                        best_score = cs
                        best_points = [p[:] for p in cur]
                elif cs >= 0:
                    d = cs - cur_score
                    if d >= 0 or rng.random() < math.exp(d / max(T, 1e-6)):
                        cur = cand
                        cur_score = cs
            T = max(T * cooling, min_T)
            if rng.random() < 0.08:
                radius = rng.choice([4, 8, 15, 30])

    return best_points

# EVOLVE-BLOCK-END
