"""Rigid-body cluster packing: fixed compact hexagonal lattice of 11 unit hexagons,
optimize global cluster (translation, rotation) + outer container orientation.
Uses a smooth log-sum-exp surrogate for the enclosing-hexagon side, refined with
a final high-precision LP polish pass."""
import math
import time
import numpy as np
import itertools

N = 11
_SQRT3 = math.sqrt(3.0)
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0

# --- Compact hexagonal cluster: center + ring1(6) + 4 from ring2 (compact selection) ---
_SCALE = 1.0
_U = (_SQRT3 * _SCALE, 0.0)
_V = (_SQRT3 / 2.0 * _SCALE, 1.5 * _SCALE)

def _build_cluster_positions():
    """Return list of (x, y) for the 11 fixed cluster hexagons (relative, zero rotation)."""
    target = [(0, 0)]
    ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    target.extend(ring1)
    ring2 = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0),
             (-1, -1), (0, -2), (1, -2), (2, -1)]
    # pick 4 ring2 that form the most compact bounding set
    best = None
    best_bbox = float('inf')
    for combo in itertools.combinations(ring2, 4):
        pts = []
        for ti, tj in list(target) + list(combo):
            x = ti * _U[0] + tj * _V[0]
            y = ti * _U[1] + tj * _V[1]
            pts.append((x, y))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bbox = (max(xs) - min(xs)) + (max(ys) - min(ys))
        if bbox < best_bbox:
            best_bbox = bbox
            best = combo
    target.extend(best)
    positions = []
    for ti, tj in target:
        x = ti * _U[0] + tj * _V[0]
        y = ti * _U[1] + tj * _V[1]
        positions.append((x, y))
    return positions

_CLUSTER_POS = _build_cluster_positions()
_CLUSTER_X = np.array([p[0] for p in _CLUSTER_POS])
_CLUSTER_Y = np.array([p[1] for p in _CLUSTER_POS])

def _cluster_vertices(theta):
    """Vertices of all 11 hexagons for cluster rotation theta (radians).
    Returns array of shape (N*6, 2)."""
    # unit hexagon vertices at rotation 0 (pointy: vertex at angle 0)
    base_angles = np.arange(6) * math.pi / 3.0
    base_cx = np.cos(base_angles)
    base_sy = np.sin(base_angles)
    c = math.cos(theta)
    s = math.sin(theta)
    # rotated vertex offsets: (x', y') = (cx*c - sy*s, cx*s + sy*c)
    rx = base_cx * c - base_sy * s
    ry = base_cx * s + base_sy * c
    # each hexagon center at _CLUSTER_X[i], _CLUSTER_Y[i]; all share rotation theta
    # vertices[i,k] = (X[i] + rx[k], Y[i] + ry[k])
    X = _CLUSTER_X[:, None] + rx[None, :]
    Y = _CLUSTER_Y[:, None] + ry[None, :]
    return np.column_stack([X.ravel(), Y.ravel()])

def _outer_normals(phi):
    """6 outward normals for outer hexagon whose vertex direction is at phi (radians).
    A regular hexagon's sides are perpendicular to directions phi + 30 + k*60."""
    base = phi + math.pi / 6.0
    angles = base + np.arange(6) * math.pi / 3.0
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _outer_side_lp(verts, phi):
    """Exact minimum enclosing hexagon side for fixed outer orientation phi.
    Returns (cx, cy, side) or None."""
    from scipy.optimize import linprog
    normals = _outer_normals(phi)
    v_dot_n = verts @ normals.T  # (M,6)
    M = len(verts)
    A_rows = []
    b_vals = []
    for k in range(6):
        nx, ny = normals[k]
        vd = v_dot_n[:, k]
        A_rows.append(np.column_stack([np.full(M, -nx), np.full(M, -ny), np.full(M, -1.0)]))
        b_vals.append(-vd)
        A_rows.append(np.column_stack([np.full(M, nx), np.full(M, ny), np.full(M, -1.0)]))
        b_vals.append(vd)
    A = np.vstack(A_rows)
    b = np.concatenate(b_vals)
    A = np.vstack([A, [[0.0, 0.0, -1.0]]])
    b = np.append(b, 0.0)
    c_obj = [0.0, 0.0, 1.0]
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs')
        if res.success:
            cx, cy, R = res.x
            return cx, cy, R / (_SQRT3 / 2.0)
    except Exception:
        pass
    return None

def _smooth_objective(params, beta=200.0):
    """Smooth log-sum-exp surrogate of enclosing-hexagon side.
    params = [cx, cy, theta_cluster, phi_outer]."""
    if time.time() > _DEADLINE - 8.0:
        return 1e9
    cx, cy, theta, phi = params
    verts = _cluster_vertices(theta)
    verts = verts + np.array([cx, cy])
    normals = _outer_normals(phi)
    v_dot_n = verts @ normals.T  # (M,6)
    # for each normal k, the constraint value R_k = max over v of (v-c).n and (c-v).n
    # (v-c).n = v.n - c.n ; (c-v).n = c.n - v.n
    c_n = np.array([cx * normals[k,0] + cy * normals[k,1] for k in range(6)])
    # R_k = max( max_v (v.n - c.n), max_v (c.n - v.n) ) = max( max_v v.n - c.n, c.n - min_v v.n )
    max_v = v_dot_n.max(axis=0)
    min_v = v_dot_n.min(axis=0)
    r1 = max_v - c_n
    r2 = c_n - min_v
    R = np.maximum(r1, r2)  # (6,)
    # smooth max via log-sum-exp
    Rmax = R.max()
    if Rmax < 0:
        Rmax = 0.0
    logsum = Rmax + math.log(np.exp((R - Rmax) * beta).sum()) / beta
    side = logsum / (_SQRT3 / 2.0)
    return float(side)

def _optimize_cluster():
    """Optimize (cx, cy, theta, phi) with L-BFGS-B on smooth surrogate, then LP polish."""
    from scipy.optimize import minimize
    best_side = float('inf')
    best_params = None
    best_center = (0.0, 0.0)
    best_theta = 0.0
    best_phi = 0.0

    # Multi-start over phi and theta
    starts = []
    for phi_deg in [0.0, 15.0, 30.0, 45.0]:
        for theta_deg in [0.0, 30.0]:
            starts.append([0.0, 0.0, math.radians(theta_deg), math.radians(phi_deg)])

    for start in starts:
        if time.time() > _DEADLINE - 15.0:
            break
        try:
            res = minimize(_smooth_objective, start, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8,
                                    'maxfun': 400})
            if res.success or res.fun < best_side:
                cx, cy, theta, phi = res.x
                # LP polish at this orientation
                verts = _cluster_vertices(theta) + np.array([cx, cy])
                lp = _outer_side_lp(verts, phi)
                if lp is not None:
                    cx_l, cy_l, side_l = lp
                    if side_l < best_side:
                        best_side = side_l
                        best_params = (cx_l, cy_l, theta, phi)
                        best_center = (cx_l, cy_l)
                        best_theta = theta
                        best_phi = phi
        except Exception:
            continue

    # Final high-precision polish: fine-sweep phi around best_phi, LP each
    if best_params is not None and time.time() < _DEADLINE - 10.0:
        verts_base = _cluster_vertices(best_theta)  # relative to cluster origin
        # re-center: we want to LP with center free, so use cluster vertices at origin
        for dphi_deg in np.arange(-2.0, 2.0 + 1e-9, 0.1):
            if time.time() > _DEADLINE - 8.0:
                break
            phi_p = best_phi + math.radians(dphi_deg)
            verts = _cluster_vertices(best_theta)  # relative
            lp = _outer_side_lp(verts, phi_p)
            if lp is not None:
                cx_l, cy_l, side_l = lp
                if side_l < best_side:
                    best_side = side_l
                    best_center = (cx_l, cy_l)
                    best_phi = phi_p

    return best_center, best_theta, best_phi, best_side

def construct():
    global _START_TIME, _DEADLINE
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    try:
        best_center, best_theta, best_phi, best_side = _optimize_cluster()
    except Exception:
        best_center = (0.0, 0.0)
        best_theta = 0.0
        best_phi = 0.0
        best_side = float('inf')

    # Build inner list from cluster positions with cluster rotation, translated by center
    cx, cy = best_center
    theta_deg = math.degrees(best_theta) % 360.0
    inner = []
    for (px, py) in _CLUSTER_POS:
        # rotate cluster point by theta, then translate by center
        c = math.cos(best_theta)
        s = math.sin(best_theta)
        X = cx + px * c - py * s
        Y = cy + px * s + py * c
        inner.append([X, Y, theta_deg])

    if best_side == float('inf'):
        # fallback: simple layout
        inner = [[_CLUSTER_POS[i][0], _CLUSTER_POS[i][1], 0.0] for i in range(N)]
        verts = _cluster_vertices(0.0)
        lp = _outer_side_lp(verts, 0.0)
        if lp is not None:
            best_center = (lp[0], lp[1])
            best_side = lp[2]
        else:
            best_side = 40.0
            best_center = (0.0, 0.0)

    return inner, best_center, best_side, math.degrees(best_phi) % 60.0
