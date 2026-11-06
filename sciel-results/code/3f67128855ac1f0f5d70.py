import time, math, random
from collections import defaultdict

def _line_key(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    g = math.gcd(abs(dx), abs(dy))
    if g:
        dx //= g
        dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _line_score(pts):
    n = len(pts)
    line_pts = defaultdict(set)
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            key = _line_key(x1, y1, x2, y2)
            line_pts[key].add(i)
            line_pts[key].add(j)
    return sum(1 for s in line_pts.values() if len(s) == 4)

def _build_pair(pts):
    pair = {}
    for i in range(16):
        for j in range(i + 1, 16):
            pair[(i, j)] = _line_key(pts[i][0], pts[i][1], pts[j][0], pts[j][1])
    return pair

def _score4(pair):
    cnt = defaultdict(int)
    for v in pair.values():
        cnt[v] += 1
    return sum(1 for v in cnt.values() if v == 4)

def _delta_move(pts, i, nx, ny, pair):
    xi, yi = pts[i]
    others = [j for j in range(16) if j != i]
    oldcnt = defaultdict(int)
    for j in others:
        k = (i, j) if i < j else (j, i)
        oldcnt[pair[k]] += 1
    newcnt = defaultdict(int)
    for j in others:
        newcnt[_line_key(xi, yi, pts[j][0], pts[j][1]) if False else _line_key(nx, ny, pts[j][0], pts[j][1])] += 1
    gain = 0
    for k in set(oldcnt) | set(newcnt):
        o = oldcnt.get(k, 0)
        nn = newcnt.get(k, 0)
        gain += (1 if nn == 4 else 0) - (1 if o == 4 else 0)
    return gain

def _apply_move(pts, i, nx, ny, pair):
    pts[i] = [nx, ny]
    for j in range(16):
        if j == i:
            continue
        k = (i, j) if i < j else (j, i)
        pair[k] = _line_key(pts[i][0], pts[i][1], pts[j][0], pts[j][1])

def _delta_swap(pts, a, b, pair):
    xa, ya = pts[a]
    xb, yb = pts[b]
    others = [j for j in range(16) if j != a and j != b]
    # old lines involving a or b
    oldcnt = defaultdict(int)
    for j in others:
        k1 = (a, j) if a < j else (j, a)
        k2 = (b, j) if b < j else (j, b)
        oldcnt[pair[k1]] += 1
        oldcnt[pair[k2]] += 1
    # line a-b
    kab = (a, b) if a < b else (b, a)
    oldcnt[pair[kab]] += 1
    # new lines: a at b's spot, b at a's spot
    newcnt = defaultdict(int)
    for j in others:
        newcnt[_line_key(xb, yb, pts[j][0], pts[j][1])] += 1
        newcnt[_line_key(xa, ya, pts[j][0], pts[j][1])] += 1
    # new a-b line (same set of points, so same line key)
    newcnt[pair[kab]] += 1
    gain = 0
    for k in set(oldcnt) | set(newcnt):
        o = oldcnt.get(k, 0)
        nn = newcnt.get(k, 0)
        gain += (1 if nn == 4 else 0) - (1 if o == 4 else 0)
    return gain

def _apply_swap(pts, a, b, pair):
    xa, ya = pts[a]
    xb, yb = pts[b]
    pts[a] = [xb, yb]
    pts[b] = [xa, ya]
    for j in range(16):
        if j == a or j == b:
            continue
        k1 = (a, j) if a < j else (j, a)
        k2 = (b, j) if b < j else (j, b)
        pair[k1] = _line_key(pts[a][0], pts[a][1], pts[j][0], pts[j][1])
        pair[k2] = _line_key(pts[b][0], pts[b][1], pts[j][0], pts[j][1])
    # a-b line unchanged (same two points)

def construct():
    start = time.time()
    deadline = start + 118.0

    def in_bounds(p):
        return abs(p[0]) <= 1000 and abs(p[1]) <= 1000

    def valid(pts):
        if len(set(tuple(p) for p in pts)) != 16:
            return False
        return all(in_bounds(p) for p in pts)

    def make_grid(cx, cy, ax, ay, mx, my):
        pts = []
        for i in range(mx):
            for j in range(my):
                pts.append([cx + i * ax, cy + j * ay])
        return pts

    def make_subset(grid_pts, k):
        idx = list(range(len(grid_pts)))
        random.shuffle(idx)
        return [list(grid_pts[idx[i]]) for i in range(k)]

    # Build initial candidate pool
    candidates = []
    # 4x4 grids (dense, known to yield many 4-lines)
    for ox in range(-3, 4):
        for oy in range(-3, 4):
            for sx in [1, 2, 3]:
                for sy in [1, 2, 3]:
                    pts = make_grid(ox, oy, sx, sy, 4, 4)
                    if valid(pts):
                        candidates.append(pts)
    # 5x5 grid subsets (denser structure, per cell finding)
    for ox in range(-3, 4):
        for oy in range(-3, 4):
            for sx in [1, 2]:
                for sy in [1, 2]:
                    grid = make_grid(ox, oy, sx, sy, 5, 5)
                    if valid(grid):
                        for _ in range(3):
                            sub = make_subset(grid, 16)
                            if valid(sub):
                                candidates.append(sub)
    # 6x5 grid subsets
    for ox in range(-3, 4):
        for oy in range(-3, 4):
            for sx in [1, 2]:
                for sy in [1, 2]:
                    grid = make_grid(ox, oy, sx, sy, 6, 5)
                    if valid(grid):
                        for _ in range(2):
                            sub = make_subset(grid, 16)
                            if valid(sub):
                                candidates.append(sub)
    # Random configs
    random.seed(42)
    for _ in range(30):
        pts = []
        for _ in range(16):
            x = random.randint(-40, 40)
            y = random.randint(-40, 40)
            pts.append([x, y])
        if valid(pts):
            candidates.append(pts)

    best = None
    best_score = -1
    for cand in candidates:
        sc = _line_score(cand)
        if sc > best_score:
            best_score = sc
            best = [list(p) for p in cand]

    # Local search with incremental delta scoring
    small_deltas = [(dx, dy) for dx in range(-5, 6) for dy in range(-5, 6) if dx != 0 or dy != 0]
    large_deltas = [(dx, dy) for dx in range(-15, 16) for dy in range(-15, 16) if dx != 0 or dy != 0]

    def local_search(pts, pair, sc, deltas, iters_per_point, allow_swaps):
        nonlocal best, best_score
        improved = True
        it = 0
        while improved and it < iters_per_point and time.time() < deadline:
            improved = False
            it += 1
            for i in range(16):
                if time.time() >= deadline:
                    return
                x, y = pts[i]
                for dx, dy in deltas:
                    nx, ny = x + dx, y + dy
                    if not in_bounds((nx, ny)):
                        continue
                    if any(nx == pts[j][0] and ny == pts[j][1] for j in range(16) if j != i):
                        continue
                    d = _delta_move(pts, i, nx, ny, pair)
                    if d > 0:
                        _apply_move(pts, i, nx, ny, pair)
                        sc += d
                        improved = True
                        if sc > best_score:
                            best_score = sc
                            best = [list(p) for p in pts]
                        x, y = nx, ny
                    elif d == 0 and random.random() < 0.1:
                        # accept neutral moves occasionally to explore
                        _apply_move(pts, i, nx, ny, pair)
                        x, y = nx, ny
            # 2-point swaps
            if allow_swaps:
                for a in range(16):
                    if time.time() >= deadline:
                        return
                    for b in range(a + 1, 16):
                        xa, ya = pts[a]
                        xb, yb = pts[b]
                        if not in_bounds((xb, yb)) or not in_bounds((xa, ya)):
                            continue
                        d = _delta_swap(pts, a, b, pair)
                        if d > 0:
                            _apply_swap(pts, a, b, pair)
                            sc += d
                            improved = True
                            if sc > best_score:
                                best_score = sc
                                best = [list(p) for p in pts]
                        elif d == 0 and random.random() < 0.05:
                            _apply_swap(pts, a, b, pair)
        return sc

    def run_anneal(init, t0, t1, iters, deltas):
        nonlocal best, best_score
        pts = [list(p) for p in init]
        pair = _build_pair(pts)
        sc = _score4(pair)
        if sc > best_score:
            best_score = sc
            best = [list(p) for p in pts]
        t = t0
        step = (t1 - t0) / max(1, iters)
        for it in range(iters):
            if time.time() >= deadline:
                break
            if random.random() < 0.7:
                # single-point move
                i = random.randrange(16)
                x, y = pts[i]
                nx = x + random.choice(deltas)[0]
                ny = y + random.choice(deltas)[1]
                if not in_bounds((nx, ny)):
                    continue
                if any(nx == pts[j][0] and ny == pts[j][1] for j in range(16) if j != i):
                    continue
                d = _delta_move(pts, i, nx, ny, pair)
                if d >= 0 or random.random() < math.exp(d / max(1e-9, t)):
                    _apply_move(pts, i, nx, ny, pair)
                    sc += d
            else:
                # swap
                a, b = random.sample(range(16), 2)
                d = _delta_swap(pts, a, b, pair)
                if d >= 0 or random.random() < math.exp(d / max(1e-9, t)):
                    _apply_swap(pts, a, b, pair)
                    sc += d
            if sc > best_score:
                best_score = sc
                best = [list(p) for p in pts]
            t = max(t1, t - step)
        return sc

    # Phase 1: local search from best candidate
    pair = _build_pair(best)
    sc = _score4(pair)
    sc = local_search(best, pair, sc, small_deltas, 8, True)

    # Phase 2: multi-start simulated annealing with large moves
    rng = random.Random(20240517)
    for trial in range(12):
        if time.time() >= deadline:
            break
        # start from best or a random perturbation
        if trial % 3 == 0:
            init = [list(p) for p in best]
        else:
            init = [list(p) for p in best]
            for _ in range(4):
                i = rng.randrange(16)
                x, y = init[i]
                nx = x + rng.randint(-20, 20)
                ny = y + rng.randint(-20, 20)
                if in_bounds((nx, ny)):
                    init[i] = [nx, ny]
        run_anneal(init, 8.0, 0.01, 3000, large_deltas)
        # polish with local search
        if time.time() < deadline:
            pair2 = _build_pair(best)
            sc2 = _score4(pair2)
            sc2 = local_search(best, pair2, sc2, small_deltas, 4, True)

    # Final polish
    if time.time() < deadline:
        pair3 = _build_pair(best)
        sc3 = _score4(pair3)
        sc3 = local_search(best, pair3, sc3, small_deltas, 6, True)

    # Ensure distinct and return
    seen = set()
    out = []
    for p in best:
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            out.append([p[0], p[1]])
    return out[:16]
