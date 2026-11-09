"""
Combined: Simulated Annealing (global) + incremental local search (polish)
+ 4-point-line repair. Score via pair-counts (line of k pts -> C(k,2) pairs)
for fast full and incremental evaluation.
"""
import time
import random
import math

_START = time.time()
_DEADLINE = _START + 115.0
_BOUND = 1000


def _line_key(p, q):
    x1, y1 = p
    x2, y2 = q
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = math.gcd(abs(dx), abs(dy))
    if g:
        dx //= g
        dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    c = x1 * dy - y1 * dx
    return (dx, dy, c)


def _count_lines(points):
    """Return (n4, n5, n3, n2) counting lines by number of points."""
    n = len(points)
    pairs = {}
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            k = _line_key(pi, points[j])
            if k is None:
                continue
            pairs[k] = pairs.get(k, 0) + 1
    n4 = n5 = n3 = n2 = 0
    for c in pairs.values():
        if c == 6:
            n4 += 1
        elif c >= 10:
            n5 += 1
        elif c == 3:
            n3 += 1
        elif c == 2:
            n2 += 1
    return n4, n5, n3, n2


def _valid(points):
    n4, n5, n3, n2 = _count_lines(points)
    return n5 == 0


def _score(points):
    n4, n5, n3, n2 = _count_lines(points)
    if n5 > 0:
        return -100000 - n5
    return n4 * 1000 + n3 * 10 + n2


def _build_ag24():
    """16 points, 16 lines of 4 (AG(2,4) structure)."""
    return [[x, y] for x in range(4) for y in range(4)]


def _seed_staggered():
    pts = []
    for row in range(4):
        for col in range(4):
            pts.append([col + row, row])
    return pts


def _seed_random_valid(rng, scale=25):
    points = []
    attempts = 0
    while len(points) < 16 and attempts < 6000:
        attempts += 1
        x = rng.randint(-scale, scale)
        y = rng.randint(-scale, scale)
        if [x, y] in points:
            continue
        cand = points + [[x, y]]
        if len(cand) >= 5 and not _valid(cand):
            continue
        points.append([x, y])
    while len(points) < 16:
        x = rng.randint(-scale, scale)
        y = rng.randint(-scale, scale)
        if [x, y] not in points:
            points.append([x, y])
    return points[:16]


def _line_structure(points):
    n = len(points)
    line_pts = {}
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            k = _line_key(pi, points[j])
            if k is None:
                continue
            s = line_pts.get(k)
            if s is None:
                line_pts[k] = {i, j}
            else:
                s.add(i)
                s.add(j)
    return line_pts


def _score_from_lines(lp):
    n4 = n5 = n3 = n2 = 0
    for s in lp.values():
        l = len(s)
        if l == 4:
            n4 += 1
        elif l >= 5:
            n5 += 1
        elif l == 3:
            n3 += 1
        elif l == 2:
            n2 += 1
    return n4, n5, n3, n2


def _apply_move(lp, i, old, new, points):
    """Update line structure after moving point i from old to new (in place)."""
    n = len(points)
    for j in range(n):
        if j == i:
            continue
        k_old = _line_key(old, points[j])
        k_new = _line_key(new, points[j])
        if k_old is not None:
            s = lp.get(k_old)
            if s is not None:
                s.discard(i)
                if not s:
                    del lp[k_old]
        if k_new is not None:
            s = lp.get(k_new)
            if s is None:
                lp[k_new] = {i, j}
            else:
                s.add(i)
                s.add(j)


def _simulated_annealing(points, rng, time_budget):
    n = len(points)
    cur = [list(p) for p in points]
    lp = _line_structure(cur)
    cur_n4, cur_n5, cur_n3, cur_n2 = _score_from_lines(lp)
    cur_score = _score(cur)
    best = [list(p) for p in cur]
    best_score = cur_score

    T = 60.0
    min_T = 0.05
    cooling = 0.999
    start = time.time()
    while time.time() < time_budget:
        if time.time() > _DEADLINE - 5.0:
            break
        i = rng.randrange(n)
        x0, y0 = cur[i]
        if rng.random() < 0.75:
            dx = rng.randint(-6, 6)
            dy = rng.randint(-6, 6)
        else:
            dx = rng.randint(-150, 150)
            dy = rng.randint(-150, 150)
        nx, ny = x0 + dx, y0 + dy
        if abs(nx) > _BOUND or abs(ny) > _BOUND:
            continue
        np_ = [nx, ny]
        if any(np_ == p for j, p in enumerate(cur) if j != i):
            continue
        old = cur[i]
        _apply_move(lp, i, old, np_, cur)
        cur[i] = np_
        n4, n5, n3, n2 = _score_from_lines(lp)
        new_score = n4 * 1000 + n3 * 10 + n2 if n5 == 0 else -100000 - n5
        if new_score >= cur_score:
            cur_score = new_score
            if new_score > best_score:
                best_score = new_score
                best = [list(p) for p in cur]
        else:
            delta = new_score - cur_score
            if rng.random() < math.exp(delta / T):
                cur_score = new_score
            else:
                _apply_move(lp, i, np_, old, cur)
                cur[i] = old
        T *= cooling
        if T < min_T:
            T = min_T
    return best, best_score


def _local_search_polish(points, rng, time_budget):
    n = len(points)
    cur = [list(p) for p in points]
    lp = _line_structure(cur)
    n4, n5, n3, n2 = _score_from_lines(lp)
    best = [list(p) for p in cur]
    best_key = (n4, -n5, n3, n2)
    step = 0
    while time.time() < time_budget:
        if time.time() > _DEADLINE - 2.0:
            break
        improved = False
        w = min(4 + step // 15, 14)
        for i in range(n):
            if time.time() >= time_budget:
                break
            x0, y0 = cur[i]
            best_local = None
            best_local_key = None
            for dx in range(-w, w + 1):
                if time.time() >= time_budget:
                    break
                for dy in range(-w, w + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x0 + dx, y0 + dy
                    if abs(nx) > _BOUND or abs(ny) > _BOUND:
                        continue
                    np_ = [nx, ny]
                    if any(np_ == p for j, p in enumerate(cur) if j != i):
                        continue
                    old = cur[i]
                    _apply_move(lp, i, old, np_, cur)
                    cur[i] = np_
                    c4, c5, c3, c2 = _score_from_lines(lp)
                    if c5 == 0:
                        key = (c4, -c5, c3, c2)
                        if best_local_key is None or key > best_local_key:
                            best_local_key = key
                            best_local = np_
                    # revert
                    _apply_move(lp, i, np_, old, cur)
                    cur[i] = old
            if best_local is not None:
                cur_key = (n4, -n5, n3, n2)
                if best_local_key > cur_key:
                    _apply_move(lp, i, cur[i], best_local, cur)
                    cur[i] = best_local
                    n4, n5, n3, n2 = _score_from_lines(lp)
                    improved = True
                    if (n4, -n5, n3, n2) > best_key:
                        best_key = (n4, -n5, n3, n2)
                        best = [list(p) for p in cur]
        step += 1
        if not improved and step > 8:
            i = rng.randrange(n)
            x0, y0 = cur[i]
            for _ in range(120):
                nx = x0 + rng.randint(-12, 12)
                ny = y0 + rng.randint(-12, 12)
                if abs(nx) > _BOUND or abs(ny) > _BOUND:
                    continue
                np_ = [nx, ny]
                if any(np_ == p for j, p in enumerate(cur) if j != i):
                    continue
                old = cur[i]
                _apply_move(lp, i, old, np_, cur)
                cur[i] = np_
                c4, c5, c3, c2 = _score_from_lines(lp)
                if c5 == 0:
                    key = (c4, -c5, c3, c2)
                    if key >= (n4, -n5, n3, n2):
                        n4, n5, n3, n2 = c4, c5, c3, c2
                        if key > best_key:
                            best_key = key
                            best = [list(p) for p in cur]
                        break
                _apply_move(lp, i, np_, old, cur)
                cur[i] = old
            step = 0
    return best, best_key


def _repair4(points, rng, time_budget):
    """Try to grow 3-point lines into 4-point lines by adding a 4th point."""
    n = len(points)
    cur = [list(p) for p in points]
    lp = _line_structure(cur)
    n4, n5, n3, n2 = _score_from_lines(lp)
    best = [list(p) for p in cur]
    best_key = (n4, -n5, n3, n2)
    while time.time() < time_budget:
        if time.time() > _DEADLINE - 2.0:
            break
        improved = False
        # candidate 3-point lines
        triples = [s for s in lp.values() if len(s) == 3]
        rng.shuffle(triples)
        for s in triples:
            if time.time() >= time_budget:
                break
            idx = list(s)
            p0, p1, p2 = cur[idx[0]], cur[idx[1]], cur[idx[2]]
            # try to find a 4th point collinear with all three
            # direction from p0->p1
            dx12, dy12 = p1[0] - p0[0], p1[1] - p0[1]
            g = math.gcd(abs(dx12), abs(dy12))
            if g:
                ux, uy = dx12 // g, dy12 // g
            else:
                continue
            # line param: p0 + t*(ux,uy). Check p2 on it
            if (p2[0] - p0[0]) * uy - (p2[1] - p0[1]) * ux != 0:
                continue
            # existing t values
            ts = set()
            for p in (p0, p1, p2):
                t = (p[0] - p0[0]) // ux if ux != 0 else (p[1] - p0[1]) // uy
                ts.add(t)
            # search nearby integer t for a new point not already in set
            base = min(ts)
            for t in range(base - 8, base + 8):
                if t in ts:
                    continue
                nx = p0[0] + t * ux
                ny = p0[1] + t * uy
                if abs(nx) > _BOUND or abs(ny) > _BOUND:
                    continue
                np_ = [nx, ny]
                if any(np_ == p for p in cur):
                    continue
                # find an index to move (move one of the triple's points? no)
                # Instead, move a point far from the line onto this spot.
                # Choose the point whose removal keeps structure best: move a
                # random point that is not on this line.
                moved = False
                for i in range(n):
                    if i in s:
                        continue
                    # moving point i to np_
                    old = cur[i]
                    _apply_move(lp, i, old, np_, cur)
                    cur[i] = np_
                    c4, c5, c3, c2 = _score_from_lines(lp)
                    if c5 == 0:
                        key = (c4, -c5, c3, c2)
                        if key > (n4, -n5, n3, n2):
                            n4, n5, n3, n2 = c4, c5, c3, c2
                            improved = True
                            if key > best_key:
                                best_key = key
                                best = [list(p) for p in cur]
                            moved = True
                            break
                    _apply_move(lp, i, np_, old, cur)
                    cur[i] = old
                if moved:
                    break
        if not improved:
            break
    return best, best_key


def construct():
    global _START, _DEADLINE
    rng = random.Random(42)
    seeds = [_build_ag24(), _seed_staggered()]
    for _ in range(6):
        seeds.append(_seed_random_valid(rng, scale=25))

    valid_seeds = [s for s in seeds if _valid(s)]
    if not valid_seeds:
        return _build_ag24()

    best_overall_pts = valid_seeds[0]
    best_overall_key = _count_lines(valid_seeds[0])

    n_seeds = len(valid_seeds)
    total_time = _DEADLINE - time.time()
    sa_time = max(2.0, (total_time * 0.45) / n_seeds)
    polish_time = max(1.0, (total_time * 0.30) / n_seeds)
    repair_time = max(0.5, (total_time * 0.20) / n_seeds)

    for s in valid_seeds:
        if time.time() >= _DEADLINE - 6.0:
            break
        t1 = min(_DEADLINE - 12.0, time.time() + sa_time)
        sa_pts, _ = _simulated_annealing(s, rng, t1)
        t2 = min(_DEADLINE - 5.0, time.time() + polish_time)
        pol_pts, _ = _local_search_polish(sa_pts, rng, t2)
        t3 = min(_DEADLINE - 2.0, time.time() + repair_time)
        rep_pts, _ = _repair4(pol_pts, rng, t3)
        key = _count_lines(rep_pts)
        if key > best_overall_key:
            best_overall_key = key
            best_overall_pts = rep_pts

    if not _valid(best_overall_pts):
        return _build_ag24()
    for p in best_overall_pts:
        if abs(p[0]) > _BOUND or abs(p[1]) > _BOUND:
            return _build_ag24()
    return [list(p) for p in best_overall_pts]
