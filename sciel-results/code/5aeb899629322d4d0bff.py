import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 6.0

def _time_ok():
    return time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN

def construct():
    if not _time_ok():
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    candidates = [
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)],
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)],
        [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)],
        [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1),(0,-1)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,1)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)],
        [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,-2)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(2,3),(1,4),(2,4)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(0,-1),(3,1)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3)],
        [(0,0),(1,0),(2,0),(3,0),(4,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(0,-1)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(1,3),(2,3)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(3,2),(3,3)],
        [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2)],
        [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(3,1),(3,2)],
    ]

    def axial_to_cartesian(axial_list):
        pts = np.array([(math.sqrt(3) * (q + r / 2.0), 1.5 * r) for q, r in axial_list])
        return pts

    def eval_exact(pts, beta, cx, cy):
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        centered = pts - np.array([cx, cy])
        projs = centered @ normals.T
        max_abs = np.max(np.abs(projs))
        S = (max_abs + APOTHEM) / APOTHEM
        return S

    def eval_smooth(pts, beta, cx, cy, temp):
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        centered = pts - np.array([cx, cy])
        projs = centered @ normals.T
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        max_val = np.max(all_projs)
        shifted = np.clip((all_projs - max_val) / temp, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temp
        D = temp * log_sum
        S = (D + APOTHEM) / APOTHEM
        return S

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = (float(np.mean(pts[:,0])), float(np.mean(pts[:,1])))

        if not _time_ok():
            return best_S, best_beta, best_center

        # Phase 1: Coarse grid over beta
        for i in range(37):
            beta = i * (math.pi / 6) / 36
            cx0 = float(np.mean(pts[:,0]))
            cy0 = float(np.mean(pts[:,1]))
            S = eval_exact(pts, beta, cx0, cy0)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = (cx0, cy0)

        if not _time_ok():
            return best_S, best_beta, best_center

        # Phase 2: Refine beta with scalar minimization (center fixed at centroid)
        try:
            def obj_beta(b):
                b = ((b % (math.pi/6)) + math.pi/6) % (math.pi/6)
                return eval_exact(pts, b, best_center[0], best_center[1])
            res = minimize_scalar(obj_beta, bounds=(0, math.pi/6), method='bounded',
                                  options={'maxiter': 200, 'xatol': 1e-12})
            if res.fun < best_S:
                best_S = res.fun
                best_beta = ((res.x % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass

        if not _time_ok():
            return best_S, best_beta, best_center

        # Phase 3: Smooth surrogate optimization over (cx, cy, beta)
        try:
            for temp in [0.3, 0.1, 0.05, 0.02]:
                if not _time_ok():
                    break
                for sb in [best_beta, best_beta + 0.05, best_beta - 0.05, 0.0, math.pi/12]:
                    if not _time_ok():
                        break
                    def smooth_obj(params, t=temp):
                        cx, cy, b = params
                        b = ((b % (math.pi/6)) + math.pi/6) % (math.pi/6)
                        return eval_smooth(pts, b, cx, cy, t)
                    init = [best_center[0], best_center[1], sb]
                    res = minimize(smooth_obj, init, method='Nelder-Mead',
                                   options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-10})
                    if res.fun < 1e10:
                        cx_r, cy_r, b_r = res.x
                        b_r = ((b_r % (math.pi/6)) + math.pi/6) % (math.pi/6)
                        S_exact = eval_exact(pts, b_r, cx_r, cy_r)
                        if S_exact < best_S:
                            best_S = S_exact
                            best_beta = b_r
                            best_center = (cx_r, cy_r)
        except Exception:
            pass

        if not _time_ok():
            return best_S, best_beta, best_center

        # Phase 4: High-precision Nelder-Mead polish with exact objective
        try:
            def exact_obj(params):
                cx, cy, b = params
                b = ((b % (math.pi/6)) + math.pi/6) % (math.pi/6)
                return eval_exact(pts, b, cx, cy)

            polish_starts = [
                [best_center[0], best_center[1], best_beta],
                [best_center[0]+0.03, best_center[1], best_beta],
                [best_center[0]-0.03, best_center[1], best_beta],
                [best_center[0], best_center[1]+0.03, best_beta],
                [best_center[0], best_center[1]-0.03, best_beta],
                [best_center[0], best_center[1], best_beta+0.02],
                [best_center[0], best_center[1], best_beta-0.02],
            ]
            for init in polish_starts:
                if not _time_ok():
                    break
                res = minimize(exact_obj, init, method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-13})
                if res.fun < best_S:
                    best_S = res.fun
                    best_center = (res.x[0], res.x[1])
                    best_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass

        return best_S, best_beta, best_center

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for axial in candidates:
        if not _time_ok():
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(best_global_pts[:,0])), float(np.mean(best_global_pts[:,1])))
        best_global_beta = 0.0
        best_global_S = eval_exact(best_global_pts, best_global_beta, best_global_center[0], best_global_center[1])

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # Scale by tiny factor to ensure strict non-overlap
    SCALE = 1.0 + 1e-12
    pts_scaled = np.array([cx + (x - cx) * SCALE, cy + (y - cy) * SCALE]).T

    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))

    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM

    inner = [[float(x), float(y), float(angle_deg)] for x, y in pts_scaled]
    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)

    return inner, center, side, rotation
