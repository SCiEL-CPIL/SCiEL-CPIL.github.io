import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 11
SQRT3 = math.sqrt(3)
DEADLINE = 150.0
SAFETY = 10.0

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _outer_hex_vertices(cx, cy, side, angle_deg):
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([side * np.cos(angles), side * np.sin(angles)])
    return vs + np.array([cx, cy])

def _outer_normals(outer_angle_deg):
    oa = np.deg2rad(outer_angle_deg)
    na = oa + np.deg2rad(30 + np.array([0, 60, 120, 180, 240, 300]))
    return np.column_stack([np.cos(na), np.sin(na)])

def _side_from_points(pts, outer_angle_deg):
    """Exact minimum enclosing regular-hexagon side for a set of points."""
    normals = _outer_normals(outer_angle_deg)
    spread = (np.max(pts @ normals.T, axis=1) - np.min(pts @ normals.T, axis=1))
    return float(np.max(spread) / SQRT3)

def _all_vertices(pos, angle_deg):
    n = len(pos)
    verts = np.zeros((n, 6, 2))
    for i in range(n):
        verts[i] = _hex_vertices(pos[i, 0], pos[i, 1], angle_deg)
    return verts

def _side(pos, angle_deg, outer_angle_deg):
    verts = _all_vertices(pos, angle_deg)
    return _side_from_points(verts.reshape(-1, 2), outer_angle_deg)

def _soft_max(vals, beta=120.0):
    m = np.max(vals)
    if m < -1e12:
        return m
    return m + np.log(np.sum(np.exp(beta * (vals - m)))) / beta

def _side_smooth(pos, angle_deg, outer_angle_deg):
    verts = _all_vertices(pos, angle_deg)
    pts = verts.reshape(-1, 2)
    normals = _outer_normals(outer_angle_deg)
    dots = pts @ normals.T
    smax = _soft_max(np.max(dots, axis=0), 120.0)
    smin = -_soft_max(-np.min(dots, axis=0), 120.0)
    return (smax - smin) / SQRT3

def _point_in_polygon(pt, poly):
    n = len(poly)
    sign = 0
    for i in range(n):
        a = poly[i]; b = poly[(i + 1) % n]
        cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if cross > 1e-12:
            if sign < 0:
                return False
            sign = 1
        elif cross < -1e-12:
            if sign > 0:
                return False
            sign = -1
    return True

def _seg_seg_dist(p1, p2, p3, p4):
    d1 = p2 - p1; d2 = p4 - p3; r = p1 - p3
    a = np.dot(d1, d1); e = np.dot(d2, d2); f = np.dot(d2, r)
    if a < 1e-12 and e < 1e-12:
        return np.linalg.norm(p1 - p3)
    if a < 1e-12:
        t = np.clip(f / e, 0, 1)
        return np.linalg.norm(p1 - (p3 + t * d2))
    c = np.dot(d1, r)
    if e < 1e-12:
        s = np.clip(c / a, 0, 1)
        return np.linalg.norm((p1 + s * d1) - p3)
    b = np.dot(d1, d2)
    denom = a * e - b * b
    if denom > 1e-12:
        s = (b * f - c * e) / denom
        s = np.clip(s, 0, 1)
        t = (b * s + f) / e
        if t < 0:
            t = 0.0; s = np.clip(c / a, 0, 1)
        elif t > 1:
            t = 1.0; s = np.clip((b + f) / a, 0, 1)
        return np.linalg.norm((p1 + s * d1) - (p3 + t * d2))
    t = np.clip(f / e, 0, 1)
    return np.linalg.norm(p1 - (p3 + t * d2))

def _poly_separation(p1, p2):
    for v in p1:
        if _point_in_polygon(v, p2):
            return 0.0
    for v in p2:
        if _point_in_polygon(v, p1):
            return 0.0
    min_dist = float('inf')
    n1 = len(p1); n2 = len(p2)
    for i in range(n1):
        a1 = p1[i]; a2 = p1[(i + 1) % n1]
        for j in range(n2):
            b1 = p2[j]; b2 = p2[(j + 1) % n2]
            d = _seg_seg_dist(a1, a2, b1, b2)
            if d < min_dist:
                min_dist = d
    return min_dist

def _overlap_penalty(pos, angle_deg, weight=200.0):
    n = len(pos)
    verts = _all_vertices(pos, angle_deg)
    pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            cd = np.linalg.norm(pos[i] - pos[j])
            if cd < 4.0:
                sep = _poly_separation(verts[i], verts[j])
                if sep < 0.005:
                    pen += weight * (0.005 - sep) ** 2
    return pen

def _objective(params, angle_deg, outer_angle_deg, n):
    pos = params.reshape(n, 2)
    return _side(pos, angle_deg, outer_angle_deg) + _overlap_penalty(pos, angle_deg)

def _objective_smooth(params, angle_deg, outer_angle_deg, n):
    pos = params.reshape(n, 2)
    return _side_smooth(pos, angle_deg, outer_angle_deg) + _overlap_penalty(pos, angle_deg)

def _is_valid(pos, angle_deg, side, outer_angle_deg, min_sep=1e-9):
    outer_v = _outer_hex_vertices(0.0, 0.0, side, outer_angle_deg)
    verts = _all_vertices(pos, angle_deg)
    for i in range(N):
        for v in verts[i]:
            if not _point_in_polygon(v, outer_v):
                return False
    for i in range(N):
        for j in range(i + 1, N):
            if _poly_separation(verts[i], verts[j]) < min_sep:
                return False
    return True

def _seed():
    eps = 1e-9
    a = SQRT3 + eps
    centers = [(0.0, 0.0)]
    for k in range(6):
        th = np.deg2rad(k * 60)
        centers.append((a * np.cos(th), a * np.sin(th)))
    for k in range(4):
        th = np.deg2rad(30 + k * 60)
        centers.append((3.0 * np.cos(th), 3.0 * np.sin(th)))
    return np.array(centers, dtype=float)

def _find_best_outer_angle(pos, angle_deg, n_coarse=24):
    verts = _all_vertices(pos, angle_deg)
    pts = verts.reshape(-1, 2)
    def S(oa):
        return _side_from_points(pts, oa)
    best_S = float('inf'); best_a = 0.0
    for i in range(n_coarse):
        ang = i * 180.0 / n_coarse
        sv = S(ang)
        if sv < best_S:
            best_S = sv; best_a = ang
    try:
        r = minimize_scalar(S, bounds=(best_a - 5, best_a + 5), method='bounded', options={'xatol': 1e-7})
        if r.fun < best_S:
            best_S = r.fun; best_a = r.x
    except Exception:
        pass
    return best_S, best_a

def _gradient_search(pos, angle_deg, outer_angle_deg, start_time, deadline, max_iter=800):
    try:
        res = minimize(_objective_smooth, pos.flatten(),
                       args=(angle_deg, outer_angle_deg, N),
                       method='L-BFGS-B',
                       options={'maxiter': max_iter, 'gtol': 1e-11, 'ftol': 1e-13, 'maxls': 40})
        if time.time() - start_time > deadline:
            return pos
        cand = res.x.reshape(N, 2)
        return cand
    except Exception:
        return pos

def _nelder(pos, angle_deg, outer_angle_deg, start_time, deadline, max_iter=400):
    try:
        res = minimize(_objective, pos.flatten(),
                       args=(angle_deg, outer_angle_deg, N),
                       method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-11, 'fatol': 1e-11})
        if time.time() - start_time > deadline:
            return pos
        return res.x.reshape(N, 2)
    except Exception:
        return pos

def _pso(pos, angle_deg, outer_angle_deg, start_time, deadline, iters=120, sw=12):
    best_g = pos.copy()
    best_g_val = _objective(best_g.flatten(), angle_deg, outer_angle_deg, N)
    P = np.zeros((sw, N, 2)); V = np.zeros((sw, N, 2))
    PB = np.zeros((sw, N, 2)); PBv = np.zeros(sw)
    for i in range(sw):
        P[i] = pos + (np.random.randn(N, 2) * 0.5 if i else np.zeros((N, 2)))
        val = _objective(P[i].flatten(), angle_deg, outer_angle_deg, N)
        PB[i] = P[i]; PBv[i] = val
        if val < best_g_val:
            best_g_val = val; best_g = P[i].copy()
    w = 0.85; c1 = 1.2; c2 = 1.2
    for _ in range(iters):
        if time.time() - start_time > deadline:
            break
        for i in range(sw):
            r1 = np.random.rand(N, 2); r2 = np.random.rand(N, 2)
            V[i] = w * V[i] + c1 * r1 * (PB[i] - P[i]) + c2 * r2 * (best_g - P[i])
            V[i] = np.clip(V[i], -0.25, 0.25)
            P[i] = P[i] + V[i]
            val = _objective(P[i].flatten(), angle_deg, outer_angle_deg, N)
            if val < PBv[i]:
                PBv[i] = val; PB[i] = P[i].copy()
                if val < best_g_val:
                    best_g_val = val; best_g = P[i].copy()
        w = max(0.5, w * 0.998)
    return best_g

def construct():
    start_time = time.time()
    deadline = DEADLINE - SAFETY
    angle = 30.0
    outer_angle = 30.0

    pos = _seed()
    best_pos = pos.copy()
    best_side = _side(pos, angle, outer_angle)

    def elapsed():
        return time.time() - start_time

    def try_accept(p):
        nonlocal best_pos, best_side
        s = _side(p, angle, outer_angle)
        if s < best_side - 1e-12 and _is_valid(p, angle, s, outer_angle):
            best_pos = p.copy(); best_side = s
            return True
        return False

    # Re-center cluster to origin (reduces enclosing side, preserves internal geometry)
    c = best_pos.mean(axis=0)
    try_accept(best_pos - c)

    # Phase 1: gradient (smooth) search
    if elapsed() < deadline - 25:
        g = _gradient_search(best_pos, angle, outer_angle, start_time, deadline - 30)
        if try_accept(g):
            pass
        # second gradient pass from a perturbed start
        if elapsed() < deadline - 20:
            g2 = _gradient_search(best_pos + np.random.randn(N, 2) * 0.15, angle, outer_angle, start_time, deadline - 25)
            try_accept(g2)

    # Phase 2: PSO exploration
    if elapsed() < deadline - 15:
        p = _pso(best_pos, angle, outer_angle, start_time, deadline - 20)
        try_accept(p)

    # Phase 3: Nelder-Mead local refinement
    if elapsed() < deadline - 10:
        nm = _nelder(best_pos, angle, outer_angle, start_time, deadline - 12)
        try_accept(nm)

    # Phase 4: fine gradient polish
    if elapsed() < deadline - 5:
        gp = _gradient_search(best_pos, angle, outer_angle, start_time, deadline - 5, max_iter=500)
        try_accept(gp)

    # Phase 5: final Nelder-Mead micro-polish
    if elapsed() < deadline - 2:
        nm2 = _nelder(best_pos, angle, outer_angle, start_time, deadline - 2, max_iter=200)
        try_accept(nm2)

    # Re-center final
    c = best_pos.mean(axis=0)
    s = _side(best_pos - c, angle, outer_angle)
    if s < best_side - 1e-12 and _is_valid(best_pos - c, angle, s, outer_angle):
        best_pos = best_pos - c
        best_side = s

    # Final validation; fall back to seed if invalid
    final_side = _side(best_pos, angle, outer_angle)
    if not _is_valid(best_pos, angle, final_side, outer_angle):
        best_pos = _seed()
        final_side = _side(best_pos, angle, outer_angle)

    inner_rows = [[float(x), float(y), float(angle)] for x, y in best_pos]
    return inner_rows, (0.0, 0.0), float(final_side), float(outer_angle)
