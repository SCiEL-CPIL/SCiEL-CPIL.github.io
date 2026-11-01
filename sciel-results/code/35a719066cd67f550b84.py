"""Packing 11 unit hexagons in a regular hexagon.
Multi-start over hexagonal lattice topologies + joint SA + high-precision Nelder-Mead.
"""
import time
import math
import random
import numpy as np
from scipy.optimize import minimize

N = 11
_T0 = time.time()
_DEADLINE = 150.0
_EPS = 1e-9

# --- geometry ---------------------------------------------------------------
# flat-top unit hexagon vertices (circumradius = 1)
_V = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])


def _verts(center, angle):
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return center + _V @ R.T


def _support(pts, dirv):
    return float(np.max(pts @ dirv))


def _containment(pts, C, S, ang):
    """Max over 6 directions of (support(pts) - support(outer)). <=0 means contained."""
    c, s = math.cos(ang), math.sin(ang)
    R = np.array([[c, -s], [s, c]])
    outer = C + _V * S
    o_dirs = (outer[:, 1:] - outer[:, :-1]).T  # 2x6
    o_sup = outer @ o_dirs
    best = -1e300
    for d in o_dirs.T:
        val = float(np.max(pts @ d)) - float(o_sup[:, d is d].max() if False else np.max(outer @ d))
        if val > best:
            best = val
    return best


def _objective(x):
    """Smooth-ish objective: containment + overlap penalties. Lower is better."""
    n = N
    centers = x[0:2 * n].reshape(n, 2)
    angles = x[2 * n:3 * n]
    C = x[3 * n:3 * n + 2]
    S = x[3 * n + 2]
    if S < 0.5:
        return 1e6
    pts = np.zeros((n * 6, 2))
    for i in range(n):
        pts[i * 6:(i + 1) * 6] = _verts(centers[i], angles[i])
    # containment
    c, s = math.cos(0.0), math.sin(0.0)  # outer angle fixed at 0 for speed in objective
    outer = C + _V * S
    o_dirs = (outer[:, 1:] - outer[:, :-1]).T
    cont = 0.0
    for j in range(6):
        d = o_dirs[:, j]
        val = float(np.max(pts @ d)) - float(np.max(outer @ d))
        if val > 0:
            cont += val * val
    # overlap: pairwise, check each inner vertex inside other hexagon
    ov = 0.0
    for i in range(n):
        pi = pts[i * 6:(i + 1) * 6]
        ci = centers[i]
        ai = angles[i]
        for j in range(i + 1, n):
            pj = pts[j * 6:(j + 1) * 6]
            cj = centers[j]
            aj = angles[j]
            # center distance quick reject
            dx = ci[0] - cj[0]; dy = ci[1] - cj[1]
            if dx * dx + dy * dy > 9.0:
                continue
            # vertex-in-hex test
            # hex j contains point p: max over 6 directions of (p - cj)·dj <= 1
            cjv = _verts(cj, aj)
            djs = (cjv[:, 1:] - cjv[:, :-1]).T
            # test pi in j
            for p in pi:
                v = p - cj
                if np.max(v @ djs) > 1.0 - _EPS:
                    ov += 1.0
                    break
            if ov > 0:
                continue
            # test pj in i
            civ = _verts(ci, ai)
            dis = (civ[:, 1:] - civ[:, :-1]).T
            for p in pj:
                v = p - ci
                if np.max(v @ dis) > 1.0 - _EPS:
                    ov += 1.0
                    break
    return S + 10.0 * cont + 5.0 * ov


def _hard_check(inner, C, S, ang):
    """Return (ok, containment_margin) using exact support/vertex tests."""
    n = len(inner)
    pts = np.zeros((n * 6, 2))
    centers = []
    for i in range(n):
        cx, cy, ad = inner[i]
        a = math.radians(ad)
        centers.append((cx, cy))
        pts[i * 6:(i + 1) * 6] = _verts((cx, cy), a)
    # containment: max over 6 outer directions
    c, s = math.cos(ang), math.sin(ang)
    R = np.array([[c, -s], [s, c]])
    outer = np.array(C) + _V * S
    o_dirs = (outer[:, 1:] - outer[:, :-1]).T
    cont = 0.0
    for j in range(6):
        d = o_dirs[:, j]
        val = float(np.max(pts @ d)) - float(np.max(outer @ d))
        if val > cont:
            cont = val
    # overlap
    for i in range(n):
        pi = pts[i * 6:(i + 1) * 6]
        ci = centers[i]
        ai = math.radians(inner[i][2])
        civ = _verts(ci, ai)
        dis = (civ[:, 1:] - civ[:, :-1]).T
        for j in range(i + 1, n):
            pj = pts[j * 6:(j + 1) * 6]
            cj = centers[j]
            aj = math.radians(inner[j][2])
            dx = ci[0] - cj[0]; dy = ci[1] - cj[1]
            if dx * dx + dy * dy > 9.0:
                continue
            cjv = _verts(cj, aj)
            djs = (cjv[:, 1:] - cjv[:, :-1]).T
            bad = False
            for p in pi:
                if np.max((p - cj) @ djs) > 1.0 - _EPS:
                    bad = True; break
            if not bad:
                for p in pj:
                    if np.max((p - ci) @ dis) > 1.0 - _EPS:
                        bad = True; break
            if bad:
                return False, cont
    return cont <= _EPS, cont


def _build_lattice(kind):
    """Return list of (x,y,angle_deg) for N unit hexagons on a lattice, plus a rough outer guess."""
    sq3 = math.sqrt(3.0)
    # center-to-center for touching flat-top hexes: horizontal 3, diagonal 1.5 & 2.598
    res = []
    if kind == 0:  # flower: 1 center + 6 ring + 4
        res.append((0.0, 0.0, 0.0))
        for k in range(6):
            a = math.radians(60 * k)
            res.append((3.0 * math.cos(a), 3.0 * math.sin(a), 0.0))
        # 4 more around
        res.append((4.5, 2.598, 0.0))
        res.append((-4.5, 2.598, 0.0))
        res.append((4.5, -2.598, 0.0))
        res.append((-4.5, -2.598, 0.0))
    elif kind == 1:  # row-based 4-3-4
        rows = [4, 3, 4]
        y = 0.0
        for r, cnt in enumerate(rows):
            x0 = -(cnt - 1) * 1.5
            for k in range(cnt):
                res.append((x0 + k * 3.0, y, 0.0))
            y += 2.598
        y0 = -2.598
        for i, (x, yy, a) in enumerate(res):
            res[i] = (x, yy + y0, a)
    elif kind == 2:  # 3-5-3
        rows = [3, 5, 3]
        y = 0.0
        for r, cnt in enumerate(rows):
            x0 = -(cnt - 1) * 1.5
            for k in range(cnt):
                res.append((x0 + k * 3.0, y, 0.0))
            y += 2.598
        y0 = -2.598
        for i, (x, yy, a) in enumerate(res):
            res[i] = (x, yy + y0, a)
    elif kind == 3:  # 5-3-3
        rows = [5, 3, 3]
        y = 0.0
        for r, cnt in enumerate(rows):
            x0 = -(cnt - 1) * 1.5
            for k in range(cnt):
                res.append((x0 + k * 3.0, y, 0.0))
            y += 2.598
        y0 = -3.897
        for i, (x, yy, a) in enumerate(res):
            res[i] = (x, yy + y0, a)
    else:  # variant: 2x5 + 1
        res = []
        for r in range(2):
            for k in range(5):
                res.append((k * 3.0 - 6.0, r * 2.598, 0.0))
        res.append((0.0, 0.0, 30.0))
    # rough outer center & side
    arr = np.array([(x, y) for x, y, a in res])
    C = arr.mean(axis=0)
    S = 12.0
    return res, C, S


def _to_vec(inner, C, S, ang):
    x = []
    for cx, cy, ad in inner:
        x += [cx, cy, math.radians(ad)]
    x += [C[0], C[1], S, ang]
    return np.array(x)


def _from_vec(x):
    n = N
    inner = []
    for i in range(n):
        inner.append((x[2 * i], x[2 * i + 1], math.degrees(x[2 * n + i])))
    C = (x[3 * n], x[3 * n + 1])
    S = x[3 * n + 2]
    ang = math.degrees(x[3 * n + 3])
    return inner, C, S, ang


def _sa_polish(x0, budget):
    n = N
    dim = 3 * n + 4
    x = x0.copy()
    f = _objective(x)
    best = x.copy(); bf = f
    T = 1.0
    step = 0.5
    rng = random.Random(12345)
    t_start = time.time()
    while time.time() - t_start < budget:
        # propose
        xi = x + np.array([rng.gauss(0, step) for _ in range(dim)])
        fi = _objective(xi)
        if fi < f or rng.random() < math.exp(-(fi - f) / max(T, 1e-6)):
            x = xi; f = fi
            if f < bf:
                bf = f; best = x.copy()
        T *= 0.985
        step = max(0.02, step * 0.99)
    return best, bf


def _nm_polish(x0, budget):
    n = N
    dim = 3 * n + 4
    t_start = time.time()
    x = x0.copy()
    f = _objective(x)
    best = x.copy(); bf = f
    # Nelder-Mead with time guard
    maxit = 400
    it = 0
    def fobj(v):
        nonlocal best, bf
        val = _objective(v)
        if val < bf:
            bf = val; best = v.copy()
        return val
    try:
        res = minimize(fobj, x, method='Nelder-Mead',
                       options={'maxiter': maxit, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        if res.x is not None and _objective(res.x) < bf:
            best = res.x.copy(); bf = _objective(res.x)
    except Exception:
        pass
    return best, bf


def construct():
    rng = random.Random(7)
    best_x = None
    best_score = 1e30
    best_result = None

    kinds = [0, 1, 2, 3, 4]
    # time allocation
    total_budget = _DEADLINE - (time.time() - _T0) - 10.0
    per_kind = max(2.0, total_budget / (len(kinds) * 2.5))

    for kind in kinds:
        if time.time() - _T0 > _DEADLINE - 12.0:
            break
        inner0, C0, S0 = _build_lattice(kind)
        x0 = _to_vec(inner0, C0, S0, 0.0)
        # SA polish
        x1, f1 = _sa_polish(x0, per_kind)
        # NM polish
        x2, f2 = _nm_polish(x1, min(per_kind, _DEADLINE - (time.time() - _T0) - 12.0))
        inner, C, S, ang = _from_vec(x2)
        ok, cont = _hard_check(inner, C, S, ang)
        if ok and S < best_score:
            best_score = S
            best_x = x2.copy()
            best_result = (inner, C, S, ang)

    # final high-precision polish on best
    if best_x is not None and time.time() - _T0 < _DEADLINE - 15.0:
        xf, ff = _nm_polish(best_x, _DEADLINE - (time.time() - _T0) - 15.0)
        inner, C, S, ang = _from_vec(xf)
        ok, cont = _hard_check(inner, C, S, ang)
        if ok and S < best_score:
            best_score = S
            best_result = (inner, C, S, ang)

    if best_result is None:
        # fallback: valid but loose
        inner0, C0, S0 = _build_lattice(0)
        return inner0, (0.0, 0.0), 14.0, 0.0
    return best_result
