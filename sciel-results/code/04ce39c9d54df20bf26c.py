"""Packing of unit regular hexagons inside a regular hexagon (minimize outer side)."""
import time
import math
import numpy as np

N = 11

def _inner_verts(cx, cy, ang_deg):
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    out = []
    for k in range(6):
        t = math.radians(60.0 * k)
        vx, vy = math.cos(t), math.sin(t)
        out.append((cx + ca * vx - sa * vy, cy + sa * vx + ca * vy))
    return out

def _outer_verts(cx, cy, s, ang_deg):
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    out = []
    for k in range(6):
        t = math.radians(60.0 * k)
        vx, vy = math.cos(t), math.sin(t)
        out.append((cx + ca * (s * vx) - sa * (s * vy),
                    cy + sa * (s * vx) + ca * (s * vy)))
    return out

def _pt_seg(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / L2
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))

def _min_edge_dist(A, B):
    d = 1e18
    for i in range(6):
        ax, ay = A[i]; bx, by = A[(i + 1) % 6]
        for j in range(6):
            cx, cy = B[j]; ex, ey = B[(j + 1) % 6]
            d = min(d, _pt_seg(ax, ay, cx, cy, ex, ey),
                    _pt_seg(cx, cy, ax, ay, bx, by))
    return d

def _inner_min_gap(pairs, verts):
    g = 1e18
    for i, j in pairs:
        g = min(g, _min_edge_dist(verts[i], verts[j]))
    return g

def _outer_min_gap(verts, ov):
    g = 1e18
    for v in verts:
        for i in range(6):
            g = min(g, _pt_seg(v[0], v[1], ov[i][0], ov[i][1], ov[(i + 1) % 6][0], ov[(i + 1) % 6][1]))
    return g

def _compute_s(inner, outer_c, ang, tol=1e-11):
    verts = [_inner_verts(x, y, a) for (x, y, a) in inner]
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    lo, hi = 1e-3, 30.0
    for _ in range(55):
        mid = 0.5 * (lo + hi)
        ov2 = _outer_verts(outer_c[0], outer_c[1], mid, ang)
        if _inner_min_gap(pairs, verts) >= -tol and _outer_min_gap(verts, ov2) >= -tol:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-9:
            break
    return hi

def _initial_layout():
    inner = [[0.0, 0.0, 0.0]]
    for k in range(6):
        a = math.radians(60.0 * k)
        inner.append([2.0 * math.cos(a), 2.0 * math.sin(a), 0.0])
    for k in range(12):
        a = math.radians(30.0 + 30.0 * k)
        inner.append([4.0 * math.cos(a), 4.0 * math.sin(a), 0.0])
    return inner

def _optimize(inner, outer_c, ang, deadline):
    inner = [[float(x), float(y), float(a)] for (x, y, a) in inner]
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    best = _compute_s(inner, outer_c, ang)
    step = 1e-2
    n = 3 * N + 2
    rng = np.random.default_rng(12345)
    for _ in range(300000):
        if time.time() > deadline:
            break
        m = 1 + (rng.random() < 0.3)
        idx = rng.integers(0, n, size=m)
        old = []
        for ii in idx:
            if ii < N:
                old.append((inner[ii][0], inner[ii][1], inner[ii][2]))
                inner[ii][0] += rng.normal(0, step)
                inner[ii][1] += rng.normal(0, step)
                inner[ii][2] += rng.normal(0, step * 0.5)
            elif ii == N:
                old.append(outer_c[0]); outer_c[0] += rng.normal(0, step * 0.5)
            else:
                old.append(outer_c[1]); outer_c[1] += rng.normal(0, step * 0.5)
        ns = _compute_s(inner, outer_c, ang)
        if ns < best - 1e-12:
            best = ns
            step = min(step * 1.05, 0.5)
        else:
            for ii, val in zip(idx, old):
                if ii < N:
                    inner[ii][0], inner[ii][1], inner[ii][2] = val
                elif ii == N:
                    outer_c[0] = val
                else:
                    outer_c[1] = val
            step = max(step * 0.9, 1e-5)
    return inner, outer_c, best

def _refine(inner, outer_c, ang, deadline):
    inner = [[float(x), float(y), float(a)] for (x, y, a) in inner]
    best = _compute_s(inner, outer_c, ang)
    step = 1e-3
    n = 3 * N + 2
    rng = np.random.default_rng(777)
    for _ in range(200000):
        if time.time() > deadline:
            break
        ii = int(rng.integers(0, n))
        old = None
        if ii < N:
            old = (inner[ii][0], inner[ii][1], inner[ii][2])
            inner[ii][0] += rng.normal(0, step)
            inner[ii][1] += rng.normal(0, step)
            inner[ii][2] += rng.normal(0, step * 0.5)
        elif ii == N:
            old = outer_c[0]; outer_c[0] += rng.normal(0, step * 0.5)
        else:
            old = outer_c[1]; outer_c[1] += rng.normal(0, step * 0.5)
        ns = _compute_s(inner, outer_c, ang)
        if ns < best - 1e-12:
            best = ns
            step = min(step * 1.02, 0.2)
        else:
            if ii < N:
                inner[ii][0], inner[ii][1], inner[ii][2] = old
            elif ii == N:
                outer_c[0] = old
            else:
                outer_c[1] = old
            step = max(step * 0.92, 1e-6)
    return inner, outer_c, best

def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    inner0 = _initial_layout()
    outer_c = [0.0, 0.0]
    ang = 0.0
    inner, outer_c, best = _optimize(inner0, outer_c, ang, deadline)
    if time.time() < deadline - 5.0:
        inner, outer_c, best = _refine(inner, outer_c, ang, deadline)
    best = _compute_s(inner, outer_c, ang)
    return inner, (float(outer_c[0]), float(outer_c[1])), float(best), float(ang)
