"""Hexagon packing: 12 unit hexagons inside a minimal regular hexagon."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 12
_START = time.time()
_DEADLINE = 120.0
_EPS = 1e-9

_NORM = np.array([
    [math.cos(math.radians(a)), math.sin(math.radians(a))]
    for a in (90.0, 30.0, 330.0, 270.0, 210.0, 150.0)
])
_VERT = np.array([
    [math.cos(math.radians(a)), math.sin(math.radians(a))]
    for a in (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)
])


def _verts(x, y, ang):
    c, s = math.cos(ang), math.sin(ang)
    R = np.array([[c, -s], [s, c]])
    return (R @ _VERT.T).T + np.array([x, y])


def _face_normals(ang):
    c, s = math.cos(ang), math.sin(ang)
    R = np.array([[c, -s], [s, c]])
    return (R @ _NORM.T).T


def _min_enclosing_hexagon(v, ang):
    Nn = _face_normals(ang)
    s = v @ Nn.T
    sup = s.max(axis=0)
    anti = (-s).max(axis=0)
    widths = np.maximum(sup[0] + anti[3], sup[3] + anti[0])
    widths = np.maximum(widths, sup[1] + anti[4])
    widths = np.maximum(widths, sup[2] + anti[5])
    side = widths.max()
    pair_of = [(0, 3), (1, 4), (2, 5)]
    best = None
    for k, m in pair_of:
        w = sup[k] + sup[m]
        if best is None or w > best[0]:
            best = (w, k, m)
    w, k, m = best
    nk = Nn[k]
    c_off = 0.5 * (sup[m] - sup[k]) * nk
    return side, (c_off[0], c_off[1])


def _objective(p):
    pos = p[:2 * N].reshape(N, 2)
    cx, cy, side, ang = p[2 * N], p[2 * N + 1], p[2 * N + 2], p[2 * N + 3]
    allv = np.vstack([_verts(x, y, 0.0) for (x, y) in pos])
    allv = allv - np.array([cx, cy])
    s, (ox, oy) = _min_enclosing_hexagon(allv, ang)
    Nn = _face_normals(ang)
    cc = np.array([cx + ox, cy + oy])
    dv = allv + np.array([cx, cy]) - cc
    sup = (dv @ Nn.T).max(axis=0)
    viol = (sup - 0.5 * s).max()
    pen = 0.0
    if viol > _EPS:
        pen = 1e6 * viol + 1e3 * viol * viol
    return s + pen


def _clip_segment(poly, a, b):
    """Clip segment a->b against convex polygon (list of (x,y) CCW)."""
    out = []
    n = len(poly)
    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i + 1) % n]
        # edge normal (outward) for CCW polygon: (dy, -dx) normalized
        ex, ey = p2[0] - p1[0], p2[1] - p1[1]
        nx, ny = ey, -ex
        def f(pt):
            return (pt[0] - p1[0]) * nx + (pt[1] - p1[1]) * ny
        fa, fb = f(a), f(b)
        ina, inb = fa >= -1e-12, fb >= -1e-12
        if ina and inb:
            out.append(b)
        elif ina and not inb:
            t = fa / (fa - fb)
            out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
        elif not ina and inb:
            t = fa / (fa - fb)
            out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
            out.append(b)
    return out


def _poly_union_area(polys):
    """Area of union of convex polygons via clipping (approx via vertex sum)."""
    # Use a simple approach: collect all vertices, clip each polygon against
    # the running union is complex; instead estimate containment via point-in-poly.
    return None


def _point_in_convex(pt, poly):
    n = len(poly)
    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i + 1) % n]
        ex, ey = p2[0] - p1[0], p2[1] - p1[1]
        nx, ny = ey, -ex
        if (pt[0] - p1[0]) * nx + (pt[1] - p1[1]) * ny < -1e-9:
            return False
    return True


def _hexagon_poly(cx, cy, side, ang):
    return [(cx + side * math.cos(math.radians(a)),
             cy + side * math.sin(math.radians(a)))
            for a in (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)]


def _exact_side(pos):
    """Exact minimal enclosing regular-hexagon side via support + containment."""
    allv = np.vstack([_verts(x, y, 0.0) for (x, y) in pos])
    best = 1e18
    for a in np.linspace(0.0, 2 * math.pi / 3.0, 6, endpoint=False):
        Nn = _face_normals(a)
        sup = (allv @ Nn.T).max(axis=0)
        anti = (-allv @ Nn.T).max(axis=0)
        widths = np.maximum(sup[0] + anti[3], sup[3] + anti[0])
        widths = np.maximum(widths, sup[1] + anti[4])
        widths = np.maximum(widths, sup[2] + anti[5])
        side = widths.max()
        # find center: midpoint of tightest pair
        pair_of = [(0, 3), (1, 4), (2, 5)]
        bestpair = None
        for k, m in pair_of:
            w = sup[k] + sup[m]
            if bestpair is None or w > bestpair[0]:
                bestpair = (w, k, m)
        w, k, m = bestpair
        nk = Nn[k]
        cx = 0.5 * (allv[:, k].max() + allv[:, k].min())
        cy = 0.5 * (allv[:, m].max() + allv[:, m].min())
        # verify containment of all vertices in hexagon(cx,cy,side,a)
        poly = _hexagon_poly(cx, cy, side, a)
        ok = all(_point_in_convex((float(v[0]), float(v[1])), poly) for v in allv)
        if ok and side < best:
            best = side
    return best


def _make_seed():
    pos = []
    dx = 1.5 * math.sqrt(3)
    dy = 1.5
    for r in range(3):
        for c in range(4):
            x = c * dx + (r % 2) * 0.75 * math.sqrt(3)
            y = (r - 1) * dy
            pos.append([x, y])
    pos = np.array(pos)
    cx, cy = pos[:, 0].mean(), pos[:, 1].mean()
    side = _exact_side(pos)
    return np.concatenate([pos.ravel(), [cx, cy, side, 0.0]])


def _refine(p):
    try:
        res = minimize(_objective, p, method='L-BFGS-B',
                       options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-10})
        return res.x, res.fun
    except Exception:
        return p, _objective(p)


def construct():
    best_p = None
    best_val = 1e18
    deadline = _START + _DEADLINE

    p0 = _make_seed()
    if time.time() > deadline - 5.0:
        pos = p0[:2 * N].reshape(N, 2)
        cx, cy, side, ang = p0[2 * N], p0[2 * N + 1], p0[2 * N + 2], p0[2 * N + 3]
        inner = [[float(x), float(y), 0.0] for (x, y) in pos]
        return inner, (float(cx), float(cy)), float(side), float(math.degrees(ang))

    p0, v0 = _refine(p0)
    if v0 < best_val:
        best_val, best_p = v0, p0

    rng = np.random.default_rng(0)
    for i in range(40):
        if time.time() > deadline - 8.0:
            break
        noise = 0.15 * (1 - i / 40.0)
        p_try = p0 + rng.normal(0.0, noise, size=p0.shape)
        p_try, v_try = _refine(p_try)
        if v_try < best_val:
            best_val, best_p = v_try, p_try

    pos = best_p[:2 * N].reshape(N, 2)
    cx, cy, side, ang = best_p[2 * N], best_p[2 * N + 1], best_p[2 * N + 2], best_p[2 * N + 3]
    try:
        side = _exact_side(pos)
    except Exception:
        pass

    inner = [[float(x), float(y), 0.0] for (x, y) in pos]
    return inner, (float(cx), float(cy)), float(side), float(math.degrees(ang))
