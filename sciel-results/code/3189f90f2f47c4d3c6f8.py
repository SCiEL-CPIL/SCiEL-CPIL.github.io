import numpy as np
from scipy.optimize import linprog
import time
from itertools import combinations

N = 11

def _hex_vertices(cx, cy, s, angle_deg):
    a = np.radians(angle_deg)
    return np.array([[cx + s * np.cos(a + i * np.pi / 3),
                      cy + s * np.sin(a + i * np.pi / 3)] for i in range(6)])

def _enclosing_hex_lp(points, angle_deg):
    a = np.radians(angle_deg)
    normals = np.array([[np.cos(a + i * np.pi / 3), np.sin(a + i * np.pi / 3)] for i in range(6)])
    projs = points @ normals.T
    max_p = np.max(projs, axis=0)
    min_p = np.min(projs, axis=0)
    sqrt3_2 = np.sqrt(3) / 2
    A_ub = []
    b_ub = []
    for k in range(6):
        A_ub.append([-normals[k, 0], -normals[k, 1], -sqrt3_2])
        b_ub.append(-max_p[k])
        A_ub.append([normals[k, 0], normals[k, 1], -sqrt3_2])
        b_ub.append(min_p[k])
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    c_obj = np.array([0.0, 0.0, 1.0])
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub,
                  bounds=[(None, None), (None, None), (0, None)], method='highs')
    if res.status != 0:
        return np.inf, 0.0, 0.0
    return res.x[2], res.x[0], res.x[1]

def _optimize_enclosing(points):
    best_S = np.inf
    best_params = None
    for ang in range(0, 60, 2):
        S, cx, cy = _enclosing_hex_lp(points, float(ang))
        if S < best_S:
            best_S = S
            best_params = (cx, cy, float(ang))
    if best_params is not None:
        best_ang = best_params[2]
        for da in np.linspace(-2, 2, 41):
            ang = (best_ang + da) % 60
            S, cx, cy = _enclosing_hex_lp(points, float(ang))
            if S < best_S:
                best_S = S
                best_params = (cx, cy, float(ang))
    return best_S, best_params

def construct():
    start_time = time.time()
    deadline = start_time + 110
    sq3 = np.sqrt(3)
    positions = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            x = q * sq3 + r * sq3 / 2
            y = r * 1.5
            positions.append((x, y))
    positions = np.array(positions)
    dists = np.sqrt(positions[:, 0] ** 2 + positions[:, 1] ** 2)
    ring0 = np.where(dists < 0.1)[0]
    ring1 = np.where((dists > 1.5) & (dists < 2.0))[0]
    ring2 = np.where((dists > 2.5) & (dists < 4.0))[0]
    best_side = np.inf
    best_result = None
    if len(ring0) >= 1 and len(ring1) >= 6 and len(ring2) >= 4:
        center_idx = ring0[0]
        ring1_idx = list(ring1[:6])
        for combo in combinations(range(len(ring2)), 4):
            if time.time() > deadline:
                break
            selected = [center_idx] + ring1_idx + [ring2[i] for i in combo]
            sel_positions = positions[selected]
            all_verts = np.vstack([_hex_vertices(p[0], p[1], 1.0, 0.0) for p in sel_positions])
            S, params = _optimize_enclosing(all_verts)
            if S < best_side:
                best_side = S
                best_result = (sel_positions.copy(), params)
    if best_result is not None:
        sel_positions, params = best_result
        cx, cy, ang = params
        inner = [[p[0], p[1], 0.0] for p in sel_positions]
        return inner, (cx, cy), best_side, ang
    inner = [[sq3 * k, 0.0, 0.0] for k in range(N)]
    return inner, (sq3 * (N - 1) / 2, 0.0), 20.0, 0.0
