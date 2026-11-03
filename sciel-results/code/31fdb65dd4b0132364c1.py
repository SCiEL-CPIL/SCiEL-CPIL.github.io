import math
import numpy as np
import time
from scipy.optimize import linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
EPS = 1e-9

def _hexagon_vertices(cx, cy, angle_deg):
    a = math.radians(angle_deg)
    vs = []
    for k in range(6):
        th = a + k * math.pi / 3.0
        vs.append((cx + math.cos(th), cy + math.sin(th)))
    return np.array(vs)

def _separating_axis_overlap(v1, v2):
    axes = []
    for k in range(6):
        dx = v1[(k + 1) % 6][0] - v1[k][0]
        dy = v1[(k + 1) % 6][1] - v1[k][1]
        L = math.hypot(dx, dy)
        if L > 0:
            axes.append((-dy / L, dx / L))
    for k in range(6):
        dx = v2[(k + 1) % 6][0] - v2[k][0]
        dy = v2[(k + 1) % 6][1] - v2[k][1]
        L = math.hypot(dx, dy)
        if L > 0:
            axes.append((-dy / L, dx / L))
    for nx, ny in axes:
        p1 = v1 @ np.array([nx, ny])
        p2 = v2 @ np.array([nx, ny])
        if p1.max() <= p2.min() or p2.max() <= p1.min():
            return 0.0, (0.0, 0.0)
    # overlap: find axis with min gap
    best_gap = float('inf')
    best_axis = (0.0, 0.0)
    for nx, ny in axes:
        p1 = v1 @ np.array([nx, ny])
        p2 = v2 @ np.array([nx, ny])
        gap = min(p1.max() - p2.min(), p2.max() - p1.min())
        if gap < best_gap:
            best_gap = gap
            best_axis = (nx, ny)
    return best_gap, best_axis

def _check_disjoint(centers, angles, tol=1e-9):
    n = len(centers)
    for i in range(n):
        v1 = _hexagon_vertices(centers[i][0], centers[i][1], angles[i])
        for j in range(i + 1, n):
            v2 = _hexagon_vertices(centers[j][0], centers[j][1], angles[j])
            gap, _ = _separating_axis_overlap(v1, v2)
            if gap < tol:
                return False
    return True

def _smallest_enclosing_hex(centers, angle_deg):
    a = math.radians(angle_deg)
    normals = []
    for k in range(6):
        ang = a + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(ang), math.sin(ang)))
    c_obj = np.array([0.0, 0.0, 1.0])
    A_ub = []
    b_ub = []
    for nx, ny in normals:
        for px, py in centers:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-(px * nx + py * ny))
            A_ub.append([nx, ny, -1.0])
            b_ub.append(px * nx + py * ny)
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(None, None), (None, None), (0, None)]
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if res.status == 0:
        return res.x[2], (res.x[0], res.x[1])
    return float('inf'), None

def _honeycomb_init():
    pts = [(0.0, 0.0)]
    used = {(0, 0)}
    dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    for _ in range(N - 1):
        added = False
        for base in list(pts):
            for d in dirs:
                np_ = (base[0] + d[0], base[1] + d[1])
                if np_ not in used:
                    q, r = np_
                    x = math.sqrt(3) * (q + r / 2.0)
                    y = 1.5 * r
                    pts.append((x, y))
                    used.add(np_)
                    added = True
                    break
            if added:
                break
    return np.array(pts)

def _cluster_init(seed):
    rng = np.random.default_rng(seed)
    pts = np.zeros((N, 2))
    pts[0] = [0.0, 0.0]
    for i in range(1, N):
        base = rng.integers(0, i)
        ang = rng.uniform(0, 2 * math.pi)
        pts[i] = pts[base] + 2.0 * np.array([math.cos(ang), math.sin(ang)])
    return pts

def _relax(centers, angles, max_iter=800):
    centers = centers.copy()
    angles = angles.copy()
    n = len(centers)
    for it in range(max_iter):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        moved = 0.0
        for i in range(n):
            vi = _hexagon_vertices(centers[i][0], centers[i][1], angles[i])
            for j in range(i + 1, n):
                vj = _hexagon_vertices(centers[j][0], centers[j][1], angles[j])
                gap, axis = _separating_axis_overlap(vi, vj)
                if gap < 1e-7:
                    # push apart
                    push = (1e-7 - gap) / 2.0 + 1e-6
                    centers[i] -= push * np.array(axis)
                    centers[j] += push * np.array(axis)
                    moved += push
        if moved < 1e-12:
            break
    return centers, angles

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    best_S = float('inf')
    best_centers = None
    best_angles = None
    best_center = None
    best_angle_deg = 0.0

    # Generate initial configurations
    inits = [_honeycomb_init()]
    for s in range(50):
        inits.append(_cluster_init(s))

    for init_pts in inits:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        centers = init_pts.copy()
        angles = np.zeros(N)
        centers, angles = _relax(centers, angles)

        # Optimize rotation + center
        best_local_S = float('inf')
        best_local_beta = 0.0
        best_local_center = None
        # Coarse grid
        for i in range(36):
            beta = i * (math.pi / 6) / 35
            D, ctr = _smallest_enclosing_hex(centers, math.degrees(beta))
            S = (D + APOTHEM) / APOTHEM
            if S < best_local_S:
                best_local_S = S
                best_local_beta = beta
                best_local_center = ctr

        # Refine rotation
        from scipy.optimize import minimize_scalar
        try:
            def obj(b):
                D, _ = _smallest_enclosing_hex(centers, math.degrees(b))
                return (D + APOTHEM) / APOTHEM
            res = minimize_scalar(obj, bounds=(0, math.pi / 6), method='bounded', options={'maxiter': 200, 'xatol': 1e-10})
            if res.fun < best_local_S:
                best_local_S = res.fun
                best_local_beta = res.x
                _, best_local_center = _smallest_enclosing_hex(centers, math.degrees(res.x))
        except Exception:
            pass

        if best_local_S < best_S:
            best_S = best_local_S
            best_centers = centers.copy()
            best_angles = angles.copy()
            best_center = best_local_center
            best_angle_deg = math.degrees(best_local_beta)

    if best_centers is None:
        # Fallback
        centers = _honeycomb_init()
        angles = np.zeros(N)
        best_angle_deg = 0.0
        D, best_center = _smallest_enclosing_hex(centers, 0.0)
        best_S = (D + APOTHEM) / APOTHEM

    # Final validity check and tiny scale-up if needed
    if not _check_disjoint(best_centers, best_angles, tol=1e-9):
        # Nudge apart
        for _ in range(100):
            for i in range(N):
                vi = _hexagon_vertices(best_centers[i][0], best_centers[i][1], best_angles[i])
                for j in range(i + 1, N):
                    vj = _hexagon_vertices(best_centers[j][0], best_centers[j][1], best_angles[j])
                    gap, axis = _separating_axis_overlap(vi, vj)
                    if gap < 1e-9:
                        push = (1e-9 - gap) / 2.0 + 1e-10
                        best_centers[i] -= push * np.array(axis)
                        best_centers[j] += push * np.array(axis)
            if _check_disjoint(best_centers, best_angles, tol=1e-9):
                break
        # Recompute enclosing hex
        D, best_center = _smallest_enclosing_hex(best_centers, best_angle_deg)
        best_S = (D + APOTHEM) / APOTHEM

    # Output
    inner = []
    for i in range(N):
        inner.append([float(best_centers[i][0]), float(best_centers[i][1]), float(best_angles[i])])
    center = [float(best_center[0]), float(best_center[1])]
    side = float(best_S)
    rotation = float(best_angle_deg)

    return inner, center, side, rotation
