import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
TIME_LIMIT = 115.0
SAFETY = 5.0

def hex_vertices(cx, cy, side, angle_deg):
    a = math.radians(angle_deg)
    return np.array([(cx + side * math.cos(a + k * math.pi / 3.0),
                      cy + side * math.sin(a + k * math.pi / 3.0)) for k in range(6)])

def min_enclosing_hexagon(verts):
    def obj(params):
        cx, cy, angle, S = params
        if S < 0.01:
            return 1e6
        apothem = S * math.sqrt(3) / 2.0
        max_viol = 0.0
        for k in range(6):
            ang = angle + math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            for px, py in verts:
                d = (px - cx) * nx + (py - cy) * ny
                if d > apothem and d - apothem > max_viol:
                    max_viol = d - apothem
        return S + max_viol * 5.0

    xs, ys = verts[:, 0], verts[:, 1]
    cx0, cy0 = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    S0 = max(xs.max() - xs.min(), ys.max() - ys.min()) * 0.7

    best_S, best_p = float('inf'), None
    for ang in [0, math.pi/12, math.pi/6, math.pi/4]:
        for dx, dy in [(-0.3, -0.3), (0, 0), (0.3, 0.3), (0.3, -0.3), (-0.3, 0.3)]:
            init = [cx0 + dx, cy0 + dy, ang, S0]
            try:
                res = minimize(obj, init, method='Nelder-Mead',
                              options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-10})
                if res.fun < best_S:
                    best_S = res.fun
                    best_p = res.x
            except Exception:
                pass

    if best_p is None:
        return S0, cx0, cy0, 0.0
    return best_p[3], best_p[0], best_p[1], math.degrees(best_p[2])

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY:
        return [[0.0, 0.0, 0.0] for _ in range(N)], [0.0, 0.0], 1.0, 0.0

    arrangements = [
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(3,2)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)],
    ]

    def axial_to_cart(axial, sf):
        return np.array([(math.sqrt(3)*(q + r/2)*sf, 1.5*r*sf) for q, r in axial])

    best_S, best_data = float('inf'), None

    for arr in arrangements:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY:
            break
        for sf in [0.85, 0.9, 0.95, 1.0, 1.05, 1.1]:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY:
                break
            centers = axial_to_cart(arr, sf)
            for hs in [0.7, 0.8, 0.9, 0.95, 1.0]:
                if time.time() - START_TIME > TIME_LIMIT - SAFETY:
                    break
                min_d = hs * math.sqrt(3)
                ok = True
                for i in range(len(centers)):
                    for j in range(i+1, len(centers)):
                        if np.linalg.norm(centers[i]-centers[j]) < min_d - 1e-9:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                verts = np.vstack([hex_vertices(cx, cy, hs, 0.0) for cx, cy in centers])
                S, ecx, ecy, eang = min_enclosing_hexagon(verts)
                if S < best_S:
                    best_S = S
                    best_data = (centers, hs, ecx, ecy, eang)

    if best_data is None:
        return [[0.0, 0.0, 0.0] for _ in range(N)], [0.0, 0.0], 1.0, 0.0

    centers, hs, ecx, ecy, eang = best_data
    inner = [[float(cx), float(cy), 0.0] for cx, cy in centers]
    return inner, [float(ecx), float(ecy)], float(best_S), float(eang)
