"""Combined program: strong AP-based seed + fast exact 4-line scoring + local search."""
import time
import random
from math import gcd

# ---------------- scoring (exact, fast) ----------------

def _norm_dir(dx, dy):
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    return (dx, dy)

def _line_key(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    d = _norm_dir(dx, dy)
    if d is None:
        return None
    a, b = d
    # line a*x + b*y = c
    c = a * p1[0] + b * p1[1]
    return (a, b, c)

def score_config(points):
    """Return (#4-point-lines, has_5collinear). Exact."""
    n = len(points)
    line_pts = {}
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            key = _line_key((xi, yi), (xj, yj))
            if key is None:
                continue
            s = line_pts.get(key)
            if s is None:
                line_pts[key] = {i, j}
            else:
                s.add(i); s.add(j)
    count4 = 0
    has5 = False
    for s in line_pts.values():
        L = len(s)
        if L == 4:
            count4 += 1
        elif L >= 5:
            has5 = True
    return count4, has5

# ---------------- 4-AP enumeration for seeding ----------------

def _enumerate_aps():
    """Return list of 4-point arithmetic progressions [p0,p1,p2,p3] with |coord|<=1000."""
    aps = []
    seen = set()
    # directions with small components
    for dx in range(-12, 13):
        for dy in range(-12, 13):
            if dx == 0 and dy == 0:
                continue
            d = _norm_dir(dx, dy)
            if d is None:
                continue
            ax, ay = d
            # enumerate starting points such that all 4 fit in [-1000,1000]
            # p0 + k*(ax,ay) for k=0..3
            # constraints:
            for x0 in range(-1000, 1001):
                # need x0, x0+ax, x0+2ax, x0+3ax in [-1000,1000]
                if ax > 0:
                    if x0 + 3 * ax > 1000 or x0 < -1000:
                        continue
                elif ax < 0:
                    if x0 + 3 * ax < -1000 or x0 > 1000:
                        continue
                for y0 in range(-1000, 1001):
                    if ay > 0:
                        if y0 + 3 * ay > 1000 or y0 < -1000:
                            continue
                    elif ay < 0:
                        if y0 + 3 * ay < -1000 or y0 > 1000:
                            continue
                    p0 = (x0, y0)
                    p1 = (x0 + ax, y0 + ay)
                    p2 = (x0 + 2 * ax, y0 + 2 * ay)
                    p3 = (x0 + 3 * ax, y0 + 3 * ay)
                    key = frozenset((p0, p1, p2, p3))
                    if key in seen:
                        continue
                    seen.add(key)
                    aps.append([list(p0), list(p1), list(p2), list(p3)])
    return aps

def _build_seed(aps):
    """Greedy: pick APs sharing at most 1 point, up to 4 APs (16 points)."""
    best = None
    rng = random.Random(7)
    candidates = aps[:]
    rng.shuffle(candidates)
    # try several greedy orders
    for order in (candidates, aps, aps[::-1]):
        chosen = []
        used = set()
        for ap in order:
            pts = set(tuple(p) for p in ap)
            inter = len(pts & used)
            if inter <= 1:
                chosen.append(ap)
                used |= pts
                if len(chosen) == 4:
                    break
        if len(chosen) == 4:
            pts = []
            for ap in chosen:
                pts.extend(ap)
            c4, has5 = score_config(pts)
            if not has5 and (best is None or c4 > best[0]):
                best = (c4, pts)
    if best is None:
        # fallback: 4x4 grid
        pts = [[x, y] for x in range(4) for y in range(4)]
        return pts
    return best[1]

# ---------------- local search ----------------

def _valid(points, i):
    pi = points[i]
    for j in range(16):
        if j != i and points[j][0] == pi[0] and points[j][1] == pi[1]:
            return False
    return True

def construct():
    t0 = time.time()
    deadline = t0 + 135  # safety margin under 150s

    # 1) build a strong seed
    aps = _enumerate_aps()
    points = _build_seed(aps)
    best_score, _ = score_config(points)
    best_points = [p[:] for p in points]

    # 2) local search
    rng = random.Random(12345)
    step_sizes = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    no_improve = 0

    while time.time() < deadline:
        improved = False
        # iterate points in a shuffled order for diversity
        order = list(range(16))
        rng.shuffle(order)
        for i in order:
            if time.time() > deadline:
                break
            orig = points[i][:]
            ox, oy = orig
            for step in step_sizes:
                if time.time() > deadline:
                    break
                # sample directions around the circle at this step
                # use a fixed set of (dx,dy) with |dx|,|dy|<=step, but limit count
                # to keep runtime bounded: try a ring
                for dx in range(-step, step + 1):
                    for dy in range(-step, step + 1):
                        if dx == 0 and dy == 0:
                            continue
                        # only test points on the "ring" to bound work
                        if abs(dx) != step and abs(dy) != step:
                            continue
                        nx = ox + dx
                        ny = oy + dy
                        if abs(nx) > 1000 or abs(ny) > 1000:
                            continue
                        points[i] = [nx, ny]
                        if not _valid(points, i):
                            points[i] = orig
                            continue
                        s, has5 = score_config(points)
                        if not has5 and s > best_score:
                            best_score = s
                            best_points = [p[:] for p in points]
                            improved = True
                            ox, oy = nx, ny  # keep moving from new position
                        else:
                            points[i] = [ox, oy]
                if time.time() > deadline:
                    break
            points[i] = [ox, oy] if not improved else points[i]

        if not improved:
            no_improve += 1
            # occasional aggressive random restart of a few points
            if no_improve >= 3 or rng.random() < 0.2:
                for _ in range(30):
                    if time.time() > deadline:
                        break
                    i = rng.randrange(16)
                    orig = points[i][:]
                    nx = rng.randint(-1000, 1000)
                    ny = rng.randint(-1000, 1000)
                    points[i] = [nx, ny]
                    if not _valid(points, i):
                        points[i] = orig
                        continue
                    s, has5 = score_config(points)
                    if not has5 and s > best_score:
                        best_score = s
                        best_points = [p[:] for p in points]
                    else:
                        points[i] = orig
            if no_improve >= 6:
                break

    # final validity check
    c4, has5 = score_config(best_points)
    if has5:
        # fallback to 4x4 grid
        return [[x, y] for x in range(4) for y in range(4)]
    return best_points
