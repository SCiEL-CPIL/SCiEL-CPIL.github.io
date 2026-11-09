import numpy as np
from scipy.optimize import minimize
import time

N = 12
START_TIME = time.time()
TIME_BUDGET = 112.0          # safety margin before the 150s kill
SPACING = 2.02               # > 2 guarantees strict disjointness of unit hexagons

def hex_vertices(cx, cy, s, theta_deg):
    theta = np.radians(theta_deg)
    angles = theta + np.arange(6) * np.pi / 3.0
    return np.stack([cx + s * np.cos(angles), cy + s * np.sin(angles)], axis=1)

def face_normals(theta_deg):
    theta = np.radians(theta_deg)
    angles = theta + np.pi / 6.0 + np.arange(6) * np.pi / 3.0
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)

def enclosing_side(vertices, c, theta_deg):
    """Minimum side of a regular hexagon (center c, rotation theta) enclosing all vertices."""
    normals = face_normals(theta_deg)
    diff = vertices - c
    proj = diff @ normals.T
    inradius = float(proj.max())
    if inradius < 0.0:
        inradius = 0.0
    return inradius / (np.sqrt(3.0) / 2.0)

def build_cluster(shape):
    e1 = np.array([SPACING, 0.0])
    e2 = np.array([SPACING * 0.5, SPACING * np.sqrt(3.0) / 2.0])
    pts = np.array([i * e1 + j * e2 for (i, j) in shape])
    pts = pts - pts.mean(axis=0)
    return pts

def optimize_enclosing(all_vertices, max_time, tight=False):
    def objective(params):
        cx, cy, theta = params
        return enclosing_side(all_vertices, np.array([cx, cy]), np.degrees(theta))
    best_side = enclosing_side(all_vertices, np.array([0.0, 0.0]), 0.0)
    best_params = np.array([0.0, 0.0, 0.0])
    t0 = time.time()
    if tight:
        starts = [best_params.copy()]
        maxiter, xatol, fatol = 30000, 1e-12, 1e-14
    else:
        starts = [np.array([cx0, cy0, np.radians(td)])
                 for td in (0, 15, 30, 45)
                 for cx0 in (-1.0, 0.0, 1.0)
                 for cy0 in (-1.0, 0.0, 1.0)]
        maxiter, xatol, fatol = 3000, 1e-9, 1e-11
    for s0 in starts:
        if time.time() - t0 > max_time:
            break
        res = minimize(objective, s0, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol})
        if res.fun < best_side:
            best_side = float(res.fun)
            best_params = np.asarray(res.x, dtype=float)
    return best_side, best_params

def construct():
    global START_TIME
    shapes = []
    shapes.append([(i, j) for i in range(3) for j in range(4)])
    shapes.append([(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(1,3),(2,0),(2,1),(2,2),(3,0),(3,1)])
    shapes.append([(0,0),(0,1),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2),(2,3),(3,0),(3,1),(3,2)])
    shapes.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(1,1),(-2,1),(-1,-1),(2,-1)])
    shapes.append([(i, j) for i in range(4) for j in range(3)])

    best_overall = None
    for shape in shapes:
        if time.time() - START_TIME > TIME_BUDGET:
            break
        pts = build_cluster(shape)
        all_vertices = np.vstack([hex_vertices(p[0], p[1], 1.0, 0.0) for p in pts])
        remaining = TIME_BUDGET - (time.time() - START_TIME)
        per_shape = max(4.0, remaining / max(1, len(shapes)))
        side, params = optimize_enclosing(all_vertices, per_shape, tight=False)
        if time.time() - START_TIME < TIME_BUDGET - 6.0:
            side2, params2 = optimize_enclosing(all_vertices, 6.0, tight=True)
            if side2 < side:
                side, params = side2, params2
        cx, cy, theta = params
        if best_overall is None or side < best_overall[0]:
            best_overall = (side, pts, (float(cx), float(cy)), float(np.degrees(theta)))

    if best_overall is None:
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

    side, pts, center, angle = best_overall
    inner = [[float(p[0]), float(p[1]), 0.0] for p in pts]
    return inner, center, float(side), angle
