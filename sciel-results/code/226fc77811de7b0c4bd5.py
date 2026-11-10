"""Packing 12 unit hexagons inside a regular hexagon.

Strategy: (1) enumerate subsets of the 19-cell side-3 hexagonal lattice,
rotating the whole cluster, and pick the 12 cells minimizing the enclosing
hexagon side; (2) polish with a fast simulated-annealing local search that
uses a cheap analytic enclosing-hexagon objective and exact SAT overlap
checks.
"""

import math
import time
import numpy as np
from itertools import combinations

try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

N = 12
SQRT3 = math.sqrt(3.0)
HALF = SQRT3 / 2.0  # apothem of a unit hexagon
START_TIME = time.time()
TIME_LIMIT = 138.0  # safety margin before 150s kill


def _hex_vertices(cx, cy, side, angle_deg):
    a = math.radians(angle_deg)
    ang = a + np.arange(6) * (math.pi / 3.0)
    return np.column_stack([cx + side * np.cos(ang),
                            cy + side * np.sin(ang)])


def _all_verts(inner):
    """inner: list of (x, y, angle_deg) -> (72,2) array of all vertices."""
    return np.vstack([_hex_vertices(x, y, 1.0, a) for x, y, a in inner])


def _enclosing_side_fast(inner, rot_deg=0.0):
    """Minimal enclosing regular hexagon (fixed rotation) via Nelder-Mead on center."""
    verts = _all_verts(inner)
    r = math.radians(rot_deg)
    normals = np.column_stack([np.cos(r + math.pi / 6 + np.arange(6) * math.pi / 3),
                               np.sin(r + math.pi / 6 + np.arange(6) * math.pi / 3)])
    proj = verts @ normals.T
    pmax = proj.max(axis=0)
    pmin = proj.min(axis=0)

    def obj(c):
        cd = np.array([c[0] * n[0] + c[1] * n[1] for n in normals])
        hw = np.maximum(pmax - cd, cd - pmin)
        return hw.max()

    if HAS_SCIPY:
        c0 = np.array([verts[:, 0].mean(), verts[:, 1].mean()])
        res = minimize(obj, c0, method='Nelder-Mead',
                       options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-9})
        cx, cy = res.x
    else:
        cx, cy = verts[:, 0].mean(), verts[:, 1].mean()
    cd = np.array([cx * n[0] + cy * n[1] for n in normals])
    hw = np.maximum(pmax - cd, cd - pmin)
    L = hw.max() / HALF
    return L, cx, cy


def _overlaps(inner, eps=1e-7):
    """SAT check for any pair of inner hexagons. Returns True if any overlap."""
    verts_list = [_hex_vertices(x, y, 1.0, a) for x, y, a in inner]
    for i in range(N):
        for j in range(i + 1, N):
            v1, v2 = verts_list[i], verts_list[j]
            sep = False
            for verts in (v1, v2):
                for k in range(6):
                    a = verts[k]; b = verts[(k + 1) % 6]
                    e = b - a
                    nrm = np.array([-e[1], e[0]])
                    nn = np.linalg.norm(nrm)
                    if nn < 1e-12:
                        continue
                    nrm /= nn
                    d1 = v1 @ nrm; d2 = v2 @ nrm
                    if d1.max() < d2.min() + eps or d2.max() < d1.min() + eps:
                        sep = True
                        break
                if sep:
                    break
            if not sep:
                return True
    return False


def _sa_loop(inner, rot, T0, T_min, cool, max_it, sigma_xy, sigma_ang, seed):
    np.random.seed(seed)
    T = T0
    cur = [list(h) for h in inner]
    cur_side, cur_cx, cur_cy = _enclosing_side_fast(cur, rot)
    best_side = cur_side
    best = [list(h) for h in cur]
    best_c = (cur_cx, cur_cy)
    for it in range(max_it):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        idx = np.random.randint(N)
        new = [list(h) for h in cur]
        new[idx][0] += np.random.normal(0, sigma_xy)
        new[idx][1] += np.random.normal(0, sigma_xy)
        new[idx][2] = (new[idx][2] + np.random.normal(0, sigma_ang)) % 60.0
        if _overlaps(new):
            continue
        ns, ncx, ncy = _enclosing_side_fast(new, rot)
        d = ns - cur_side
        if d < 0 or np.random.random() < math.exp(-d / max(T, 1e-12)):
            cur = new; cur_side = ns; cur_cx = ncx; cur_cy = ncy
            if cur_side < best_side:
                best_side = cur_side
                best = [list(h) for h in cur]
                best_c = (cur_cx, cur_cy)
        T = max(T * cool, T_min)
    return best, best_c, best_side


def construct():
    global START_TIME
    START_TIME = time.time()
    deadline = START_TIME + TIME_LIMIT

    # 19 centers of the side-3 hexagonal lattice (flat-top axial coords)
    all_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                x = 1.5 * q
                y = SQRT3 * (r + q / 2.0)
                all_centers.append((x, y))

    # Outer-hexagon edge normals for a flat-top regular hexagon
    normal_angles = [math.radians(30 + 60 * j) for j in range(6)]
    normal_cos = np.array([math.cos(a) for a in normal_angles])
    normal_sin = np.array([math.sin(a) for a in normal_angles])

    def _L_for_hex(cx, cy, theta):
        """Smallest outer side L for a unit hex at (cx,cy) rot theta, flat-top outer."""
        va = np.array([math.radians(60 * k) + theta for k in range(6)])
        vx = cx + np.cos(va)
        vy = cy + np.sin(va)
        projs = vx[:, None] * normal_cos[None, :] + vy[:, None] * normal_sin[None, :]
        return projs.max() / HALF

    def _optimal_theta(cx, cy):
        """Rotation theta in [0,60) minimizing _L_for_hex."""
        best_L = float('inf')
        best_t = 0.0
        for t_deg in range(0, 61, 3):
            t = math.radians(t_deg)
            L = _L_for_hex(cx, cy, t)
            if L < best_L:
                best_L = L
                best_t = t
        for step in (1.0, 0.1, 0.01):
            lo = max(0.0, best_t - step * 5)
            hi = min(math.radians(60), best_t + step * 5)
            for t_deg in np.arange(lo, hi + step, step):
                t = math.radians(t_deg)
                L = _L_for_hex(cx, cy, t)
                if L < best_L:
                    best_L = L
                    best_t = t
        return best_L, best_t

    SCALE = 1.0000001
    best_L = float('inf')
    best_subset = None
    best_thetas = None
    best_cluster_angle = 0.0

    # Coarse + key cluster rotations (6-fold symmetry -> 0..60)
    cluster_angles = sorted(set([math.radians(a) for a in
                                 list(range(0, 61, 5)) + [15, 30, 45, 52.5, 22.5, 7.5]]))

    for cluster_angle in cluster_angles:
        if time.time() > deadline - 20.0:
            break
        ca = math.cos(cluster_angle)
        sa = math.sin(cluster_angle)
        rot_centers = [(x * ca - y * sa, x * sa + y * ca) for x, y in all_centers]

        center_opt = []
        for i in range(19):
            x, y = rot_centers[i]
            L_h, th_h = _optimal_theta(x * SCALE, y * SCALE)
            center_opt.append((L_h, th_h))

        # Pruning: sort by L; if the 12th-smallest is >= best_L, skip this rotation
        order = sorted(range(19), key=lambda i: center_opt[i][0])
        if center_opt[order[11]][0] >= best_L:
            continue

        for subset in combinations(range(19), 12):
            if time.time() > deadline - 20.0:
                break
            L = max(center_opt[i][0] for i in subset)
            if L < best_L:
                best_L = L
                best_subset = subset
                best_thetas = [center_opt[i][1] for i in subset]
                best_cluster_angle = cluster_angle
                if best_L < 9.9:
                    break

    if best_subset is None:
        # Fallback: first 12 cells, no cluster rotation
        best_cluster_angle = 0.0
        best_subset = tuple(range(12))
        best_thetas = [0.0] * 12

    # Build initial positions from best discrete solution
    ca = math.cos(best_cluster_angle)
    sa = math.sin(best_cluster_angle)
    init_inner = []
    for idx, i in enumerate(best_subset):
        x, y = all_centers[i]
        rx = (x * ca - y * sa) * SCALE
        ry = (x * sa + y * ca) * SCALE
        init_inner.append([rx, ry, math.degrees(best_thetas[idx])])

    # Choose outer rotation (0 or 30) that fits the cluster best
    best_rot = 0.0
    best_init_side = float('inf')
    for rot in (0.0, 30.0):
        if time.time() > deadline - 15.0:
            break
        s, cx, cy = _enclosing_side_fast(init_inner, rot)
        if s < best_init_side:
            best_init_side = s
            best_rot = rot

    # SA polish (coarse -> fine), guarding the deadline
    cur_inner, cur_c, cur_side = init_inner, None, best_init_side
    if time.time() < deadline - 10.0:
        b, c, s = _sa_loop(cur_inner, best_rot, 1.0, 0.02, 0.995, 2500, 0.3, 5.0, 42)
        if s < cur_side:
            cur_inner, cur_c, cur_side = b, c, s
    if time.time() < deadline - 8.0:
        b, c, s = _sa_loop(cur_inner, best_rot, 0.5, 0.01, 0.997, 1500, 0.2, 3.0, 123)
        if s < cur_side:
            cur_inner, cur_c, cur_side = b, c, s
    if time.time() < deadline - 5.0:
        b, c, s = _sa_loop(cur_inner, best_rot, 0.05, 0.001, 0.99, 1200, 0.05, 1.0, 777)
        if s < cur_side:
            cur_inner, cur_c, cur_side = b, c, s

    # Final high-precision enclosing hexagon
    final_side, final_cx, final_cy = _enclosing_side_fast(cur_inner, best_rot)

    return cur_inner, (final_cx, final_cy), final_side, best_rot
