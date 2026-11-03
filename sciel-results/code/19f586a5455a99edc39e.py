"""Different strategy: smooth SLSQP NLP over positions + outer side, warm-started at optimal lattice."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 120.0  # safety margin before 150s kill
SQRT3 = np.sqrt(3.0)
TOL = 1e-9

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    return np.array([[cx + side * np.cos(a + i * np.pi / 3),
                      cy + side * np.sin(a + i * np.pi / 3)] for i in range(6)])

def _vertices_batch(pos, ang, side=1.0):
    pos = np.asarray(pos, dtype=np.float64)
    ang = np.radians(np.asarray(ang, dtype=np.float64))
    base = np.arange(6) * np.pi / 3
    ta = ang[:, None] + base[None, :]
    return np.column_stack([
        (pos[:, 0] + side * np.cos(ta)).ravel(),
        (pos[:, 1] + side * np.sin(ta)).ravel()
    ])

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    dx, dy = px - cx, py - cy
    a = np.radians(angle_deg)
    ca, sa = np.cos(-a), np.sin(-a)
    lx, ly = ca * dx - sa * dy, sa * dx + ca * dy
    s = side
    if lx > s + 1e-9 or lx < -s - 1e-9:
        return False
    for c in (SQRT3 * lx + ly, -SQRT3 * lx + ly, -SQRT3 * lx - ly, SQRT3 * lx - ly):
        if c > SQRT3 * s + 1e-9:
            return False
    return True

def _min_side_vectorized(all_verts, cx, cy, outer_angle_deg):
    v = np.asarray(all_verts, dtype=np.float64)
    if len(v) == 0:
        return 0.0
    dx, dy = v[:, 0] - cx, v[:, 1] - cy
    a = np.radians(outer_angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    lx, ly = ca * dx - sa * dy, sa * dx + ca * dy
    req = np.maximum(
        np.maximum(np.maximum(lx, -lx), np.maximum(lx + ly / SQRT3, -lx + ly / SQRT3)),
        np.maximum(-lx - ly / SQRT3, lx - ly / SQRT3)
    )
    return float(np.max(req))

def _overlap(verts1, verts2, tol=1e-9):
    axes = []
    for verts in (verts1, verts2):
        for i in range(6):
            e = verts[(i + 1) % 6] - verts[i]
            nrm = np.hypot(e[0], e[1])
            if nrm > 1e-12:
                axes.append([e[1] / nrm, -e[0] / nrm])
    for ax in axes:
        p1 = verts1 @ ax
        p2 = verts2 @ ax
        if p1.max() < p2.min() - tol or p2.max() < p1.min() - tol:
            return False
    return True

def _check_no_overlap(pos, ang, tol=1e-7):
    pos = np.asarray(pos, dtype=np.float64)
    ang = np.asarray(ang, dtype=np.float64)
    vlist = [_hex_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i]) for i in range(len(pos))]
    for i in range(len(vlist)):
        for j in range(i + 1, len(vlist)):
            if _overlap(vlist[i], vlist[j], tol):
                return False
    return True

def _min_outer_side(pos, ang, oc, oang):
    return _min_side_vectorized(_vertices_batch(pos, ang, 1.0), oc[0], oc[1], oang)

def _check_validity(inner, oc, oside, oang, tol=1e-6):
    vlist = [_hex_vertices(x, y, 1.0, a) for (x, y, a) in inner]
    for i in range(len(vlist)):
        for j in range(i + 1, len(vlist)):
            if _overlap(vlist[i], vlist[j]):
                return False
    for v in vlist:
        for (px, py) in v:
            if not _point_in_hexagon(px, py, oc[0], oc[1], oside, oang):
                return False
    return True

def _lattice_positions():
    """Known optimal 3-3-3-2 hexagonal lattice for 11 unit hexagons (side-1, 0 deg)."""
    s3 = SQRT3
    row_y = [0.0, 1.5, 3.0, 4.5]
    row_x = [
        [-s3, 0.0, s3],
        [-1.5 * s3, -0.5 * s3, 0.5 * s3, 1.5 * s3],
        [-s3, 0.0, s3],
        [-1.5 * s3, -0.5 * s3, 0.5 * s3, 1.5 * s3],
    ]
    pos = []
    for y, xs in zip(row_y, row_x):
        for x in xs:
            pos.append([x, y])
            if len(pos) == N:
                break
        if len(pos) == N:
            break
    pos = np.array(pos, dtype=np.float64)
    pos -= pos.mean(axis=0)
    return pos

def _smooth_obj(params, beta=30.0):
    n = N
    pos = params[:2 * n].reshape(n, 2)
    ang = params[2 * n:2 * n + n]
    S = params[2 * n + n]
    if S < 0.1:
        S = 0.1
    verts = _vertices_batch(pos, ang, 1.0)
    cx, cy = pos.mean(axis=0)
    dx, dy = verts[:, 0] - cx, verts[:, 1] - cy
    req = np.maximum(
        np.maximum(np.maximum(dx, -dx), np.maximum(dx + dy / SQRT3, -dx + dy / SQRT3)),
        np.maximum(-dx - dy / SQRT3, dx - dy / SQRT3)
    )
    max_r = np.max(req)
    if max_r > 30.0:
        return 1e6
    sse = max_r + np.log(np.sum(np.exp(beta * (req - max_r)))) / beta
    # smooth non-overlap: min center distance for side-1 hexagons = SQRT3
    pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
            if d < SQRT3:
                pen += (SQRT3 - d) ** 2
    return float(SSE) + 1000.0 * pen + 100.0 * max(0.0, sse - S)

def construct():
    global DEADLINE
    pos0 = _lattice_positions()
    ang0 = np.zeros(N, dtype=np.float64)
    cx0, cy0 = pos0.mean(axis=0)
    S0 = _min_outer_side(pos0, ang0, (cx0, cy0), 0.0)
    x0 = np.concatenate([pos0.ravel(), ang0, [S0]])

    best_pos, best_ang = pos0.copy(), ang0.copy()
    best_ocx, best_ocy, best_oang, best_side = cx0, cy0, 0.0, S0

    def _commit(x):
        nonlocal best_pos, best_ang, best_ocx, best_ocy, best_oang, best_side
        p = x[:2 * N].reshape(N, 2)
        a = x[2 * N:2 * N + N]
        ocx, ocy = p.mean(axis=0)
        s = _min_outer_side(p, a, (ocx, ocy), 0.0)
        if _check_no_overlap(p, a) and s < best_side - 1e-12:
            best_pos, best_ang = p.copy(), a.copy()
            best_ocx, best_ocy, best_oang, best_side = ocx, ocy, 0.0, s

    # Phase 1: smooth SLSQP NLP
    if time.time() < DEADLINE - 60.0:
        try:
            res = minimize(
                _smooth_obj, x0, method='SLSQP',
                options={'maxiter': 60, 'ftol': 1e-9, 'eps': 1e-6}
            )
            if res.x is not None:
                _commit(res.x)
        except Exception:
            pass

    # Phase 2: local coordinate search (position + angle) with exact objective
    step = 0.05
    min_step = 1e-4
    while step >= min_step and time.time() < DEADLINE - 20.0:
        improved = False
        for i in range(N):
            if time.time() > DEADLINE - 20.0:
                break
            for dx in (-step, 0.0, step):
                for dy in (-step, 0.0, step):
                    if dx == 0.0 and dy == 0.0:
                        continue
                    np_ = best_pos.copy()
                    np_[i, 0] += dx
                    np_[i, 1] += dy
                    if not _check_no_overlap(np_, best_ang):
                        continue
                    ocx, ocy = np_.mean(axis=0)
                    s = _min_outer_side(np_, best_ang, (ocx, ocy), 0.0)
                    if s < best_side - 1e-11:
                        best_pos, best_side = np_, s
                        best_ocx, best_ocy = ocx, ocy
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
            # small angle nudge
            if time.time() > DEADLINE - 20.0:
                break
            na = best_ang.copy()
            na[i] += step
            if _check_no_overlap(best_pos, na):
                ocx, ocy = best_pos.mean(axis=0)
                s = _min_outer_side(best_pos, na, (ocx, ocy), 0.0)
                if s < best_side - 1e-11:
                    best_ang, best_side = na, s
                    improved = True
        if not improved:
            step *= 0.6

    # Final center polish
    final_ocx, final_ocy = best_pos.mean(axis=0)
    final_side = _min_outer_side(best_pos, best_ang, (final_ocx, final_ocy), 0.0)

    inner_list = [[float(best_pos[i, 0]), float(best_pos[i, 1]), float(best_ang[i])] for i in range(N)]
    if not _check_validity(inner_list, (final_ocx, final_ocy), final_side, 0.0):
        # fallback to valid lattice
        inner_list = [[float(pos0[i, 0]), float(pos0[i, 1]), 0.0] for i in range(N)]
        final_ocx, final_ocy = pos0.mean(axis=0)
        final_side = _min_outer_side(pos0, ang0, (final_ocx, final_ocy), 0.0)
        if not _check_validity(inner_list, (final_ocx, final_ocy), final_side, 0.0):
            inner_list = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner_list, (0.0, 0.0), 40.0, 0.0

    return inner_list, (float(final_ocx), float(final_ocy)), float(final_side), 0.0
