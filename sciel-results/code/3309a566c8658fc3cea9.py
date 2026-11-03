import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11
START = time.time()
DEADLINE = 150.0
SQRT3 = math.sqrt(3.0)
APOTHEM = SQRT3 / 2.0   # apothem of a unit hexagon (inradius)
EPS = 1e-9


def _hex_vertices(cx, cy, s, rot):
    """Vertices of a regular hexagon (center (cx,cy), side s, rotation rot in deg)."""
    a0 = math.radians(rot)
    pts = []
    for k in range(6):
        a = a0 + math.radians(60 * k)
        pts.append((cx + s * math.cos(a), cy + s * math.sin(a)))
    return pts


def _inner_hex(cx, cy, ang_deg):
    """Unit regular hexagon (side 1) with center (cx,cy) and rotation ang_deg."""
    a0 = math.radians(ang_deg)
    pts = []
    for k in range(6):
        a = a0 + math.radians(60 * k)
        pts.append((cx + math.cos(a), cy + math.sin(a)))
    return pts


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _point_in_poly(pt, poly):
    n = len(poly)
    sign = 0
    for i in range(n):
        c = _cross(poly[i], poly[(i + 1) % n], pt)
        if abs(c) < 1e-9:
            continue
        s = 1 if c > 0 else -1
        if sign == 0:
            sign = s
        elif sign != s:
            return False
    return True


def _poly_overlap(A, B):
    """SAT: convex polygons overlap iff no separating axis exists."""
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            ax = poly[(i + 1) % n][0] - poly[i][0]
            ay = poly[(i + 1) % n][1] - poly[i][1]
            nx, ny = -ay, ax
            minA = min(p[0] * nx + p[1] * ny for p in A)
            maxA = max(p[0] * nx + p[1] * ny for p in A)
            minB = min(p[0] * nx + p[1] * ny for p in B)
            maxB = max(p[0] * nx + p[1] * ny for p in B)
            if maxA + 1e-9 < minB or maxB + 1e-9 < minA:
                return False
    return True


def _in_outer(inner_pts, outer_pts):
    return all(_point_in_poly(p, outer_pts) for p in inner_pts)


def _valid(inner, outer_c, s, rot):
    outer_pts = _hex_vertices(outer_c[0], outer_c[1], s, rot)
    polys = [_inner_hex(x, y, a) for (x, y, a) in inner]
    for i in range(len(polys)):
        if not _in_outer(polys[i], outer_pts):
            return False
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if _poly_overlap(polys[i], polys[j]):
                return False
    return True


def _min_outer_side(inner, outer_c, rot):
    """Exact minimal side length of outer hexagon (center outer_c, rotation rot)
    that contains every inner hexagon.  Uses support functions of convex polygons."""
    a0 = math.radians(rot)
    dirs = [(math.cos(a0 + math.radians(60 * k)),
             math.sin(a0 + math.radians(60 * k))) for k in range(6)]
    polys = [_inner_hex(x, y, a) for (x, y, a) in inner]
    maxd = 0.0
    for poly in polys:
        for dx, dy in dirs:
            for sx, sy in (-1.0, 1.0):
                proj = max((px * sx * dx + py * sy * dy) for px, py in poly)
                d = (proj - (outer_c[0] * sx * dx + outer_c[1] * sy * dy)) / APOTHEM
                if d > maxd:
                    maxd = d
    return maxd


def _penalty_vec(inner, oc, rot):
    """Vector of per-hexagon containment margins (positive = violates outer of side 1)."""
    a0 = math.radians(rot)
    dirs = [(math.cos(a0 + math.radians(60 * k)),
             math.sin(a0 + math.radians(60 * k))) for k in range(6)]
    vals = np.empty(N)
    for i in range(N):
        x, y, ang = inner[i]
        a0i = math.radians(ang)
        verts = [(x + math.cos(a0i + math.radians(60 * k)),
                  y + math.sin(a0i + math.radians(60 * k))) for k in range(6)]
        m = 0.0
        for dx, dy in dirs:
            for sx, sy in (-1.0, 1.0):
                proj = max(px * sx * dx + py * sy * dy for px, py in verts)
                d = (proj - (oc[0] * sx * dx + oc[1] * sy * dy)) / APOTHEM
                if d > m:
                    m = d
        vals[i] = m
    return vals


def _smooth_penalty(inner, oc, rot):
    """Differentiable-ish objective: minimize max containment margin + overlap penalty."""
    v = _penalty_vec(inner, oc, rot)
    m = float(np.max(v))
    # smooth-ish max term
    t = 10.0
    sm = math.logsumexp(t * v) / t if hasattr(math, 'logsumexp') else m
    # overlap penalty via SAT on all pairs
    polys = [_inner_hex(x, y, a) for (x, y, a) in inner]
    pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            A, B = polys[i], polys[j]
            for poly in (A, B):
                n = len(poly)
                sep = None
                for kk in range(n):
                    ax = poly[(kk + 1) % n][0] - poly[kk][0]
                    ay = poly[(kk + 1) % n][1] - poly[kk][1]
                    nx, ny = -ay, ax
                    minA = min(p[0] * nx + p[1] * ny for p in A)
                    maxA = max(p[0] * nx + p[1] * ny for p in A)
                    minB = min(p[0] * nx + p[1] * ny for p in B)
                    maxB = max(p[0] * nx + p[1] * ny for p in B)
                    gap = min(minA - maxB, minB - maxA)  # negative if overlap
                    if gap > 0:
                        break
                    if sep is None or gap > sep:
                        sep = gap
                if sep is None:
                    break
            if sep is None:
                # overlapping: add penalty proportional to (1 - best_gap)
                pen += 1.0
            elif sep < 1e-4:
                pen += (1e-4 - sep) * 5.0
    return m + 2.0 * pen


def _hexagonal_seed():
    """Deterministic hexagonal lattice of N unit hexagons (angle 0)."""
    dx, dy = 3.0, SQRT3
    rows = {}
    r = 0
    while len(rows) < N:
        rows[r] = []
        r += 1
    # fill a centered hex-shaped cluster
    placed = []
    # generate candidate lattice points and pick N closest to origin
    cands = []
    for i in range(-6, 7):
        for j in range(-6, 7):
            x = i * dx + (j % 2) * 1.5
            y = j * dy
            cands.append((x, y, math.hypot(x, y)))
    cands.sort(key=lambda c: c[2])
    for x, y, _ in cands[:N]:
        placed.append([x, y, 0.0])
    return placed, (0.0, 0.0), 0.0


def _random_seed(rng):
    placed = [[rng.uniform(-4, 4), rng.uniform(-4, 4), rng.uniform(0, 60)] for _ in range(N)]
    return placed, (0.0, 0.0), 0.0


def _objective(inner, oc, s, rot):
    return 0.0 if _valid(inner, oc, s, rot) else 1.0


def _refine(inner, oc, rot, rng, budget):
    """Nelder-Mead on smooth penalty, then binary-search shrink of side."""
    def x0():
        v = np.empty(3 * N + 2)
        for i in range(N):
            v[3 * i] = inner[i][0]
            v[3 * i + 1] = inner[i][1]
            v[3 * i + 2] = inner[i][2]
        v[-2] = oc[0]
        v[-1] = oc[1]
        return v

    def obj(v):
        if time.time() > budget:
            return 1e6
        inner_v = []
        for i in range(N):
            inner_v.append([v[3 * i], v[3 * i + 1], v[3 * i + 2]])
        return _smooth_penalty(inner_v, (v[-2], v[-1]), rot)

    try:
        res = minimize(obj, x0(), method='Nelder-Mead',
                       options={'maxiter': 600, 'xatol': 1e-4, 'fatol': 1e-5})
        v = res.x
        inner_v = [[v[3 * i], v[3 * i + 1], v[3 * i + 2]] for i in range(N)]
        oc = (v[-2], v[-1])
    except Exception:
        inner_v = inner
        oc = oc

    # now compute exact minimal side
    s = _min_outer_side(inner_v, oc, rot)
    return inner_v, oc, s, rot


def _shrink(inner, oc, rot, budget):
    """Binary search the exact minimal side (validity is monotone in s)."""
    s = _min_outer_side(inner, oc, rot)
    lo, hi = 0.0, s
    for _ in range(40):
        if time.time() > budget:
            break
        mid = 0.5 * (lo + hi)
        if _valid(inner, oc, mid, rot):
            hi = mid
        else:
            lo = mid
    return inner, oc, hi, rot


def construct():
    budget = START + DEADLINE - 3.0
    rng = random.Random(20240517)
    best = None
    best_s = float('inf')

    def consider(inner, oc, s, rot):
        nonlocal best, best_s
        if _valid(inner, oc, s, rot) and s < best_s:
            best_s = s
            best = (inner, oc, s, rot)

    # deterministic lattice seed (primary)
    if time.time() < budget - 30.0:
        inner, oc, rot = _hexagonal_seed()
        inner, oc, s, rot = _refine(inner, oc, rot, rng, budget - 5.0)
        inner, oc, s, rot = _shrink(inner, oc, rot, budget - 2.0)
        consider(inner, oc, s, rot)

    # a few rotated lattice variants
    for r in (10.0, 30.0, 5.0):
        if time.time() > budget - 25.0:
            break
        inner, oc, rot0 = _hexagonal_seed()
        inner, oc, s, rot = _refine(inner, oc, rot0, rng, budget - 8.0)
        inner, oc, s, rot = _shrink(inner, oc, rot, budget - 2.0)
        consider(inner, oc, s, rot)

    # random restarts with refinement
    nrestart = 0
    while time.time() < budget - 8.0 and nrestart < 25:
        nrestart += 1
        inner, oc, rot = _random_seed(rng)
        inner, oc, s, rot = _refine(inner, oc, rot, rng, budget - 6.0)
        inner, oc, s, rot = _shrink(inner, oc, rot, budget - 2.0)
        consider(inner, oc, s, rot)

    if best is None:
        # fallback: large valid box
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 60.0, 0.0

    inner, oc, s, rot = best
    return inner, (oc[0], oc[1]), s, rot
