import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hex_vertices(cx, cy, s, angle_deg):
    theta = np.radians(angle_deg)
    return np.array([[cx + s * np.cos(theta + k * np.pi / 3),
                      cy + s * np.sin(theta + k * np.pi / 3)] for k in range(6)])

def min_outer_side_for_inner(inner_hexes, ox, oy, oang_deg):
    all_verts = []
    for (cx, cy, ang) in inner_hexes:
        verts = hex_vertices(cx, cy, 1.0, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    theta = np.radians(oang_deg)
    apothem_factor = np.sqrt(3) / 2
    max_val = 0.0
    for k in range(6):
        a = theta + np.radians(30 + k * 60)
        vals = (all_verts[:, 0] - ox) * np.cos(a) + (all_verts[:, 1] - oy) * np.sin(a)
        max_val = max(max_val, np.max(vals))
    
    return max_val / apothem_factor

def full_objective(params):
    inner = []
    for i in range(N):
        cx = params[3 * i]
        cy = params[3 * i + 1]
        ang = params[3 * i + 2]
        inner.append((cx, cy, ang))
    ox, oy, oang = params[36], params[37], params[38]
    return min_outer_side_for_inner(inner, ox, oy, oang)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    sqrt3 = np.sqrt(3)

    # Topology 1: 1 center + 6 ring + 5 partial ring
    t1 = [(0.0, 0.0)]
    for k in range(6):
        a = np.radians(60 * k)
        t1.append((sqrt3 * np.cos(a), sqrt3 * np.sin(a)))
    for k in range(5):
        a = np.radians(60 * k + 30)
        t1.append((2 * sqrt3 * np.cos(a), 2 * sqrt3 * np.sin(a)))

    # Topology 2: 3 rows x 4 cols
    t2 = []
    for row in range(3):
        for col in range(4):
            x = col * sqrt3
            y = row * 1.5
            if row % 2 == 1:
                x += sqrt3 / 2
            t2.append((x, y))

    # Topology 3: 4 rows x 3 cols
    t3 = []
    for row in range(4):
        for col in range(3):
            x = col * sqrt3
            y = row * 1.5
            if row % 2 == 1:
                x += sqrt3 / 2
            t3.append((x, y))

    # Topology 4: 2 rows x 6 cols
    t4 = []
    for row in range(2):
        for col in range(6):
            x = col * sqrt3
            y = row * 1.5
            if row % 2 == 1:
                x += sqrt3 / 2
            t4.append((x, y))

    topologies = [t1, t2, t3, t4]

    best_side = np.inf
    best_params = None

    for topo in topologies:
        if time.time() > deadline - 20:
            break

        for angle_pattern in [0, 1]:
            if time.time() > deadline - 20:
                break

            init = []
            for i, (x, y) in enumerate(topo):
                if angle_pattern == 0:
                    ang = 0.0
                else:
                    ang = 0.0 if i % 2 == 0 else 30.0
                init.extend([x, y, ang])
            init.extend([0.0, 0.0, 0.0])

            try:
                res = minimize(full_objective, init, method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-4, 'fatol': 1e-5, 'adaptive': True})
                if res.fun < best_side:
                    best_side = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass

    # Random restarts
    if time.time() < deadline - 40:
        for trial in range(3):
            if time.time() > deadline - 20:
                break
            np.random.seed(100 + trial)
            init = []
            for i in range(N):
                x = np.random.uniform(-4, 4)
                y = np.random.uniform(-4, 4)
                ang = np.random.uniform(0, 60)
                init.extend([x, y, ang])
            init.extend([0.0, 0.0, 0.0])
            try:
                res = minimize(full_objective, init, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-3, 'fatol': 1e-4})
                if res.fun < best_side:
                    best_side = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass

    if best_params is not None:
        inner = [[best_params[3 * i], best_params[3 * i + 1], best_params[3 * i + 2]] for i in range(N)]
        ox, oy = best_params[36], best_params[37]
        oang = best_params[38]
    else:
        topo = topologies[0]
        inner = [[x, y, 0.0] for (x, y) in topo]
        all_verts = np.vstack([hex_vertices(cx, cy, 1.0, ang) for (cx, cy, ang) in inner])
        ox, oy = float(np.mean(all_verts[:, 0])), float(np.mean(all_verts[:, 1]))
        oang = 0.0
        best_side = min_outer_side_for_inner(inner, ox, oy, oang)

    return inner, (ox, oy), float(best_side), float(oang)
