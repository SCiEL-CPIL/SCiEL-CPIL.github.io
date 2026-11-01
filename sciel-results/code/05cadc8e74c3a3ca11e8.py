import math
import random
import time
import numpy as np

N = 11
DEADLINE = 120.0
SAFE_MARGIN = 8.0
RNG = random.Random(20240517)

# ----------------------------------------------------------------------
# Geometry helpers (unit regular hexagon, flat-top orientation)
# ----------------------------------------------------------------------
def hex_vertices(angle_deg, cx=0.0, cy=0.0):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for i in range(6):
        t = math.radians(60.0 * i)
        x = math.cos(t)
        y = math.sin(t)
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts

def convex_hull(points):
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-12:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-12:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

def circumradius(poly):
    cx = sum(p[0] for p in poly) / len(poly)
    cy = sum(p[1] for p in poly) / len(poly)
    return max(math.hypot(p[0] - cx, p[1] - cy) for p in poly)

def side_from_circumradius(r):
    return r / math.cos(math.pi / 6.0)

def _all_vertices(inner):
    verts = []
    for (x, y, ang) in inner:
        verts.extend(hex_vertices(ang, x, y))
    return verts

def _min_side(inner):
    verts = _all_vertices(inner)
    return side_from_circumradius(circumradius(convex_hull(verts)))

def _hard_side(inner, eps=1e-9):
    verts = _all_vertices(inner)
    return side_from_circumradius(circumradius(convex_hull(verts))) + eps

# ----------------------------------------------------------------------
# Objective: smooth container + soft pairwise disjointness penalty
# ----------------------------------------------------------------------
def _pairwise_violation(inner, lam=1.0):
    n = len(inner)
    v = 0.0
    for i in range(n):
        vi = hex_vertices(inner[i][2], inner[i][0], inner[i][1])
        for j in range(i + 1, n):
            vj = hex_vertices(inner[j][2], inner[j][0], inner[j][1])
            for a in vi:
                for b in vj:
                    d = math.hypot(a[0] - b[0], a[1] - b[1])
                    if d < lam:
                        v += (lam - d) ** 2
    return v

def _objective(inner, lam=0.4):
    return _min_side(inner) + 50.0 * _pairwise_violation(inner, lam)

# ----------------------------------------------------------------------
# Simulated annealing over (x, y, angle) of each hexagon
# ----------------------------------------------------------------------
def _sa(inner, t0, tmin, steps, max_step, lam=0.4):
    cur = [list(h) for h in inner]
    cur_c = _min_side(cur)
    cur_v = _pairwise_violation(cur, lam)
    cur_f = cur_c + 50.0 * cur_v
    best = [list(h) for h in cur]
    best_c = cur_c
    best_f = cur_f
    n = len(cur)
    for s in range(steps):
        if time.time() > DEADLINE - SAFE_MARGIN:
            break
        i = RNG.randrange(n)
        cur[i][0] += (RNG.random() * 2 - 1) * max_step
        cur[i][1] += (RNG.random() * 2 - 1) * max_step
        cur[i][2] += (RNG.random() * 2 - 1) * 12.0
        nc = _min_side(cur)
        nv = _pairwise_violation(cur, lam)
        nf = nc + 50.0 * nv
        if nf < cur_f or RNG.random() < math.exp(-(nf - cur_f) / max(t0 * (tmin / t0) ** (s / max(steps - 1, 1)), 1e-9)):
            cur_f, cur_c, cur_v = nf, nc, nv
            if nc < best_c:
                best_c = nc
                best = [list(h) for h in cur]
                best_f = nf
    return best, best_c

# ----------------------------------------------------------------------
# Nelder-Mead local polish on the smooth objective
# ----------------------------------------------------------------------
def _nm(inner, iters=250):
    x = np.array([c for h in inner for c in h], dtype=float)
    n = x.size
    def f(v):
        inn = [tuple(v[3 * i:3 * i + 3]) for i in range(N)]
        return _objective(inn)
    simplex = [x.copy()]
    for i in range(n):
        p = x.copy()
        p[i] += 0.5 if abs(x[i]) < 1e-6 else 0.5 * x[i]
        simplex.append(p)
    simplex.sort(key=f)
    for _ in range(iters):
        if time.time() > DEADLINE - SAFE_MARGIN:
            break
        fvals = [f(s) for s in simplex]
        centroid = np.mean(simplex[:-1], axis=0)
        # reflect
        xr = centroid + 1.0 * (centroid - simplex[-1])
        fr = f(xr)
        if fr < fvals[0]:
            xe = centroid + 2.0 * (centroid - simplex[-1])
            fe = f(xe)
            simplex[-1] = xe if fe < fr else xr
        elif fr < fvals[-2]:
            simplex[-1] = xr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
            else:
                for k in range(1, n + 1):
                    simplex[k] = 0.5 * (simplex[k] + simplex[0])
        simplex.sort(key=f)
    inn = [tuple(x[3 * i:3 * i + 3]) for i in range(N)]
    return inn, _min_side(inn)

# ----------------------------------------------------------------------
# Exact local polish: push hexagons apart while shrinking the container
# ----------------------------------------------------------------------
def _exact_polish(inner, iters=400):
    pos = [list(h) for h in inner]
    # current hard radius
    def rad():
        return circumradius(convex_hull(_all_vertices(pos)))
    r = rad()
    best = [list(h) for h in pos]
    best_r = r
    for _ in range(iters):
        if time.time() > DEADLINE - SAFE_MARGIN:
            break
        improved = False
        # shrink container by moving each hexagon toward centroid
        cx = sum(p[0] for p in pos) / N
        cy = sum(p[1] for p in pos) / N
        for i in range(N):
            for step in (0.02, 0.01, 0.005, 0.002):
                dx = (cx - pos[i][0]) * step
                dy = (cy - pos[i][1]) * step
                nx, ny = pos[i][0] + dx, pos[i][1] + dy
                pos[i][0], pos[i][1] = nx, ny
                r2 = rad()
                # check no overlap with others
                ok = True
                vi = hex_vertices(pos[i][2], nx, ny)
                for j in range(N):
                    if j == i:
                        continue
                    vj = hex_vertices(pos[j][2], pos[j][0], pos[j][1])
                    for a in vi:
                        for b in vj:
                            if math.hypot(a[0] - b[0], a[1] - b[1]) < 1e-9:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if ok and r2 < r:
                    r = r2
                    improved = True
                    break
                else:
                    pos[i][0], pos[i][1] = nx - dx, ny - dy
        if r < best_r:
            best_r = r
            best = [list(h) for h in pos]
        if not improved:
            break
    return best, best_r

# ----------------------------------------------------------------------
# Main search
# ----------------------------------------------------------------------
def construct():
    t_start = time.time()
    best = None
    best_side = float('inf')

    def consider(inner):
        nonlocal best, best_side
        if time.time() > DEADLINE - SAFE_MARGIN:
            return
        c = _hard_side(inner)
        if c < best_side:
            best_side = c
            best = [list(h) for h in inner]

    # ---- initial seeds ----
    seeds = []
    # concentric flower
    flower = [[0.0, 0.0, 0.0]]
    for k in range(6):
        a = math.radians(60.0 * k)
        flower.append([2.0 * math.cos(a), 2.0 * math.sin(a), 0.0])
    seeds.append(flower)
    # linear row
    row = [[2.0 * k, 0.0, 0.0] for k in range(-5, 6)]
    seeds.append(row)
    # random
    for _ in range(3):
        seeds.append([[RNG.uniform(-4, 4), RNG.uniform(-4, 4), RNG.uniform(0, 180)] for _ in range(N)])

    for seed in seeds:
        if time.time() > DEADLINE - SAFE_MARGIN:
            break
        # simulated annealing
        sa_res, _ = _sa(seed, 1.5, 0.02, 1500, 0.6, lam=0.4)
        consider(sa_res)
        # Nelder-Mead polish
        nm_res, _ = _nm(sa_res)
        consider(nm_res)
        # exact polish
        ex_res, _ = _exact_polish(nm_res)
        consider(ex_res)

    if best is None:
        best = [list(h) for h in seeds[0]]
        best_side = _hard_side(best)

    # final safety: verify disjointness; if any overlap, fall back to a known-good
    # (flower) arrangement which is disjoint.
    def is_disjoint(inner):
        for i in range(N):
            vi = hex_vertices(inner[i][2], inner[i][0], inner[i][1])
            for j in range(i + 1, N):
                vj = hex_vertices(inner[j][2], inner[j][0], inner[j][1])
                for a in vi:
                    for b in vj:
                        if math.hypot(a[0] - b[0], a[1] - b[1]) < 1e-9:
                            return False
        return True

    if not is_disjoint(best):
        best = [list(h) for h in seeds[0]]
        best_side = _hard_side(best)

    inner = [[float(x), float(y), float(a)] for (x, y, a) in best]
    return inner, (0.0, 0.0), float(best_side), 0.0
