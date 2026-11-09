"""Construct 20 integer points (|x|,|y|<=100), no four collinear, minimizing the
largest 3-collinear-free subset. Uses exact branch-and-bound for the score and a
time-guarded search over line-based configurations."""
import time
import random
from itertools import combinations

_DEADLINE = time.time() + 140.0  # safety margin under the 150s kill
_rng = random.Random(20240517)


def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def _line_points(px, py, dx, dy, k):
    """k distinct integer points on the line through (px,py) with primitive direction (dx,dy)."""
    pts = []
    seen = set()
    for t in range(k):
        x = px + dx * t
        y = py + dy * t
        if -100 <= x <= 100 and -100 <= y <= 100 and (x, y) not in seen:
            pts.append((x, y))
            seen.add((x, y))
    return pts


def _make_config(lines):
    """Build 20 distinct in-bounds points from a list of (px,py,dx,dy,k) line specs."""
    pts = []
    used = set()
    for (px, py, dx, dy, k) in lines:
        got = 0
        t = 0
        while got < k and t < 400:
            x = px + dx * t
            y = py + dy * t
            if -100 <= x <= 100 and -100 <= y <= 100 and (x, y) not in used:
                pts.append((x, y))
                used.add((x, y))
                got += 1
            t += 1
    return pts


def _max_no_three_collinear(points):
    """Exact largest subset with no three collinear via branch and bound."""
    n = len(points)
    if n <= 2:
        return n
    triples = [set(c) for c in combinations(range(n), 3) if _collinear3(points, c)]
    if not triples:
        return n
    # index each point by the triples it participates in
    pt_tr = [set() for _ in range(n)]
    for i, tr in enumerate(triples):
        for p in tr:
            pt_tr[p].add(i)
    # order points by descending triple-degree (most constrained first)
    order = sorted(range(n), key=lambda p: -len(pt_tr[p]))
    best = [0]
    chosen = [False] * n
    tri_hit = [False] * len(triples)

    def bound(cur, idx):
        # upper bound: cur + remaining points (trivial) but prune if can't beat best
        return cur + (n - idx)

    def rec(cur, idx):
        if cur + (n - idx) <= best[0]:
            return
        if idx == n:
            if cur > best[0]:
                best[0] = cur
            return
        p = order[idx]
        # try to include p if it doesn't complete a triple
        if not any(tri_hit[i] for i in pt_tr[p]):
            newly = [i for i in pt_tr[p] if not tri_hit[i]]
            for i in newly:
                tri_hit[i] = True
            rec(cur + 1, idx + 1)
            for i in newly:
                tri_hit[i] = False
        # try to exclude p
        rec(cur, idx + 1)

    rec(0, 0)
    return best[0]


def _collinear3(points, c):
    (x1, y1), (x2, y2), (x3, y3) = points[c[0]], points[c[1]], points[c[2]]
    return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0


def _has_four_collinear(points):
    n = len(points)
    for a, b, c, d in combinations(range(n), 4):
        (x1, y1), (x2, y2) = points[a], points[b]
        if (x2 - x1) * (points[c][1] - y1) - (y2 - y1) * (points[c][0] - x1) != 0:
            continue
        if (x2 - x1) * (points[d][1] - y1) - (y2 - y1) * (points[d][0] - x1) != 0:
            continue
        return True
    return False


def _score_and_valid(points):
    if len(points) != 20 or len(set(map(tuple, points))) != 20:
        return None
    if any(abs(x) > 100 or abs(y) > 100 for x, y in points):
        return None
    if _has_four_collinear(points):
        return None
    return _max_no_three_collinear(points)


def _line_specs_random():
    """Random list of line specs covering 20 points: one big line + one medium + smalls."""
    specs = []
    # big line: k=11
    dx, dy = _rng.choice([(1, 0), (0, 1), (1, 1), (1, -1), (2, 1), (1, 2), (3, 1)])
    px = _rng.randint(-100 + dx * 10, 100 - dx * 10) if dx else _rng.randint(-89, 89)
    py = _rng.randint(-100 + dy * 10, 100 - dy * 10) if dy else _rng.randint(-89, 89)
    specs.append((px, py, dx, dy, 11))
    # medium line: k=6, different slope
    dx2, dy2 = _rng.choice([(1, 0), (0, 1), (1, 1), (1, -1), (2, 1), (1, 2), (3, 2)])
    px2 = _rng.randint(-95, 95)
    py2 = _rng.randint(-95, 95)
    specs.append((px2, py2, dx2, dy2, 6))
    # fillers as short lines / singles
    remaining = 20 - 11 - 6
    for _ in range(remaining):
        dx3, dy3 = _rng.choice([(1, 0), (0, 1), (1, 1), (1, -1), (2, 1), (1, 2)])
        px3 = _rng.randint(-99, 99)
        py3 = _rng.randint(-99, 99)
        specs.append((px3, py3, dx3, dy3, 1))
    return specs


def _local_improve(points):
    """Try swapping individual points to reduce the score."""
    pts = [list(p) for p in points]
    cur = _score_and_valid(pts)
    if cur is None:
        return None
    improved = True
    while improved and time.time() < _DEADLINE - 5:
        improved = False
        for i in range(20):
            for _ in range(6):
                cand = [p[:] for p in pts]
                cand[i] = [_rng.randint(-100, 100), _rng.randint(-100, 100)]
                s = _score_and_valid(cand)
                if s is not None and s < cur:
                    pts = cand
                    cur = s
                    improved = True
                    break
            if improved:
                break
    return cur


def construct():
    start = time.time()
    # baseline: guaranteed valid config with score 4 (two lines 11 + 9)
    base_lines = [
        (0, 0, 1, 0, 11),    # 11 points on x-axis: (0..10, 0)
        (0, 100, 1, 0, 9),   # 9 points: (0..8, 100)
    ]
    base_pts = _make_config(base_lines)
    best_score = _score_and_valid(base_pts)
    if best_score is None:
        best_score = 20
        base_pts = [[x, x * x] for x in range(-10, 10)]

    # search for better configs under time guard
    best = (best_score, base_pts)
    while time.time() < _DEADLINE - 10:
        specs = _line_specs_random()
        pts = _make_config(specs)
        s = _score_and_valid(pts)
        if s is not None and s < best[0]:
            best = (s, pts)
            # try to improve locally
            s2 = _local_improve(pts)
            if s2 is not None and s2 < best[0]:
                best = (s2, pts)
        if best[0] <= 3:
            break

    return [list(p) for p in best[1]]


if __name__ == "__main__":
    print(construct())
