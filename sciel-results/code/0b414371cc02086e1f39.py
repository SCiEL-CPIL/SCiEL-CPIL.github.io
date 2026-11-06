import time
import random
from math import gcd

DEADLINE = 138.0  # safety margin before 150 s kill


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
    """Return (lines, line_masks). lines: list of frozensets of collinear indices (size>=3)."""
    n = len(points)
    line_map = {}
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            key = _line_key(pi, points[j])
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
    return lines, line_masks


def _has_four(points, lines):
    for L in lines:
        if len(L) >= 4:
            return True
    return False


def _max_cap_free(points, lines, line_masks, time_limit=5.0):
    """Exact largest subset with no three collinear, via branch and bound.

    A subset S is 3-collinear-free iff no line (size>=3) has all its points in S.
    We decide points in order; a line becomes 'bad' once it has 2 points in S and a
    3rd is still undecided -> the 3rd must be excluded (forced).  Upper bound =
    current + remaining points + points on undecided lines (any valid extension is a
    subset of that union), which is tight enough to prune heavily.
    """
    n = len(points)
    line_count = len(lines)
    start = time.time()
    timed_out = [False]

    point_lines = [[] for _ in range(n)]
    for li, L in enumerate(lines):
        for idx in L:
            point_lines[idx].append(li)
    line_size = [len(L) for L in lines]
    line_mask = list(line_masks)

    state = [0] * line_count   # 0 undecided, 1 has>=2 in S (bad if a 3rd enters), 2 bad (>=3 in S)
    best = [0]
    cur = [0]

    def bound(idx):
        rem = n - idx
        extra = 0
        for li in range(line_count):
            if state[li] == 0:
                extra += line_size[li]
        return cur[0] + rem + extra

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
        # ---- include p ----
        ok = True
        touched = []
        for li in point_lines[p]:
            st = state[li]
            if st == 2:
                ok = False
                break
            if st == 1:
                state[li] = 2
                touched.append(li)
        if ok:
            cur[0] += 1
            dfs(idx + 1)
            cur[0] -= 1
            for li in touched:
                state[li] = 1
        if timed_out[0]:
            return
        # ---- exclude p ----
        ok = True
        touched = []
        for li in point_lines[p]:
            if state[li] == 1:
                state[li] = 0
                touched.append(li)
        if ok:
            dfs(idx + 1)
            for li in touched:
                state[li] = 1

    dfs(0)
    return best[0]


def _score(points, time_limit=5.0):
    lines, lm = _build_lines(points)
    if _has_four(points, lines):
        return 999
    return _max_cap_free(points, lines, lm, time_limit)


def _valid(points):
    if len(set(tuple(p) for p in points)) != 20:
        return False
    if any(abs(p[0]) > 100 or abs(p[1]) > 100 for p in points):
        return False
    lines, lm = _build_lines(points)
    return not _has_four(points, lines)


def _parabola():
    return [[x, x * x // 4] for x in range(-10, 10)]


def _candidates():
    cands = [_parabola()]
    # Hand-designed: parabola y=x^2 with a few points moved to form small collinear
    # triples (score ~17-19).  These seed the local search.
    for dx, dy in [(-100, -100), (100, 100), (100, -100), (-100, 100),
                   (50, 80), (-60, 40), (70, -30), (-40, -70)]:
        c = _parabola()
        c[0] = [c[0][0] + dx, c[0][1] + dy]
        c[19] = [c[19][0] + dx, c[19][1] + dy]
        cands.append(c)
    return cands


def construct():
    t0 = time.time()
    best_points = _parabola()
    best_score = 20

    def consider(pts, tl):
        nonlocal best_points, best_score
        if not _valid(pts):
            return
        lines, lm = _build_lines(pts)
        s = _max_cap_free(pts, lines, lm, tl)
        if s < best_score:
            best_score = s
            best_points = [p[:] for p in pts]

    # 1) seed candidates
    for c in _candidates():
        if time.time() - t0 > DEADLINE - 20:
            break
        consider(c, 4.0)

    # 2) local search from the best config
    cur = [p[:] for p in best_points]
    rng = random.Random(12345)
    while time.time() - t0 < DEADLINE - 5:
        improved = False
        a = rng.randrange(20)
        ox, oy = cur[a]
        for _ in range(24):
            if time.time() - t0 > DEADLINE - 5:
                break
            nx = rng.randint(max(-100, ox - 6), min(100, ox + 6))
            ny = rng.randint(max(-100, oy - 6), min(100, oy + 6))
            cur[a] = [nx, ny]
            if len(set(tuple(p) for p in cur)) != 20:
                cur[a] = [ox, oy]
                continue
            lines, lm = _build_lines(cur)
            if _has_four(cur, lines):
                cur[a] = [ox, oy]
                continue
            s = _max_cap_free(cur, lines, lm, 2.0)
            if s < best_score:
                best_score = s
                best_points = [p[:] for p in cur]
                improved = True
            else:
                cur[a] = [ox, oy]
        if not improved and rng.random() < 0.3:
            continue
        # occasional 2-swap
        if time.time() - t0 < DEADLINE - 8:
            a2, b2 = rng.sample(range(20), 2)
            oa, ob = cur[a2][:], cur[b2][:]
            for _ in range(4):
                if time.time() - t0 > DEADLINE - 8:
                    break
                na = [rng.randint(max(-100, oa[0] - 5), min(100, oa[0] + 5)),
                      rng.randint(max(-100, oa[1] - 5), min(100, oa[1] + 5))]
                nb = [rng.randint(max(-100, ob[0] - 5), min(100, ob[0] + 5)),
                      rng.randint(max(-100, ob[1] - 5), min(100, ob[1] + 5))]
                cur[a2], cur[b2] = na, nb
                if len(set(tuple(p) for p in cur)) != 20:
                    cur[a2], cur[b2] = oa, ob
                    continue
                lines, lm = _build_lines(cur)
                if _has_four(cur, lines):
                    cur[a2], cur[b2] = oa, ob
                    continue
                s = _max_cap_free(cur, lines, lm, 2.0)
                if s < best_score:
                    best_score = s
                    best_points = [p[:] for p in cur]
                else:
                    cur[a2], cur[b2] = oa, ob

    if not _valid(best_points):
        return _parabola()
    return best_points
