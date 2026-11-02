import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar, linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def axial_to_cartesian(axial_list):
        pts = np.empty((len(axial_list), 2), dtype=float)
        for i, (q, r) in enumerate(axial_list):
            pts[i, 0] = math.sqrt(3) * (q + r / 2.0)
            pts[i, 1] = 1.5 * r
        return pts

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
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,-2)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(2,3),(1,4),(2,4)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(0,-1),(3,1)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(1,4),(2,4)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3)],
        [(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2)],
        [(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)],
    ]

    def calc_S_lp(pts, beta):
        """Exact minimum enclosing hexagon via linear programming."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None

        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))

        n_pts = len(pts)
        # Variables: cx, cy, D (apothem of enclosing hexagon)
        # Minimize D
        # Subject to: for each normal n_k and each point p_i:
        #   (p_i - c) . n_k <= D  =>  -cx*nk_x - cy*nk_y - D <= -p_i . n_k
        #   -(p_i - c) . n_k <= D =>  cx*nk_x + cy*nk_y - D <= p_i . n_k

        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = np.empty((2 * n_pts * 6, 3))
        b_ub = np.empty(2 * n_pts * 6)

        idx = 0
        for nx, ny in normals:
            for px, py in pts:
                dot = px * nx + py * ny
                A_ub[idx, :] = [-nx, -ny, -1.0]
                b_ub[idx] = -dot
                idx += 1
                A_ub[idx, :] = [nx, ny, -1.0]
                b_ub[idx] = dot
                idx += 1

        bounds = [(None, None), (None, None), (0.0, None)]

        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = float(res.x[2])
                cx = float(res.x[0])
                cy = float(res.x[1])
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def calc_S_direct(pts, beta, cx=None, cy=None):
        """Direct computation of enclosing hexagon size given center and rotation."""
        if cx is None or cy is None:
            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))
        max_proj = 0.0
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            nx = math.cos(ang)
            ny = math.sin(ang)
            projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
            max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
        S = (max_proj + APOTHEM) / APOTHEM
        return S, (cx, cy), max_proj

    def smooth_calc_S(pts, beta, cx, cy, temperature=0.1):
        """Smooth surrogate using log-sum-exp."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        max_val = float(np.max(all_projs))
        if max_val > 100:
            return float('inf')
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = float(np.log(np.sum(np.exp(shifted)))) + max_val / temperature
        D = temperature * log_sum
        S = (D + APOTHEM) / APOTHEM
        return S

    def optimize_shape(pts):
        """Optimize rotation and center for a given point configuration."""
        best_S = float('inf')
        best_beta = 0.0
        best_center = None

        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None

        # Phase 1: Grid search over rotation using exact LP
        n_grid = 61
        for i in range(n_grid):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            beta = i * (math.pi / 6) / (n_grid - 1)
            S, center, D = calc_S_lp(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center

        if best_center is None:
            return float('inf'), 0.0, None

        # Phase 2: Refine rotation with minimize_scalar using LP
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def obj_beta(b):
                    S, _, _ = calc_S_lp(pts, b)
                    return S
                res = minimize_scalar(obj_beta, bounds=(0, math.pi / 6),
                                      method='bounded', options={'maxiter': 200, 'xatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = float(res.x)
                    _, best_center, _ = calc_S_lp(pts, best_beta)
        except Exception:
            pass

        # Phase 3: Smooth multi-start optimization over (cx, cy, beta)
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                temperatures = [0.5, 0.2, 0.1, 0.05, 0.02]
                start_betas = [best_beta, best_beta + 0.03, best_beta - 0.03, 0.0, math.pi / 12, math.pi / 6]

                for temp in temperatures:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    for sb in start_betas:
                        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                            break

                        def smooth_obj(params, t=temp):
                            cx, cy, beta = params
                            beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                            return smooth_calc_S(pts, beta, cx, cy, temperature=t)

                        init = [best_center[0], best_center[1], sb]
                        res = minimize(smooth_obj, init, method='Nelder-Mead',
                                       options={'maxiter': 600, 'xatol': 1e-12, 'fatol': 1e-12})

                        if res.fun < 1e10:
                            cx_r, cy_r, beta_r = res.x
                            beta_r = ((beta_r % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                            # Verify with exact LP
                            S_exact, c_exact, _ = calc_S_lp(pts, beta_r)
                            if S_exact < best_S:
                                best_S = S_exact
                                best_beta = beta_r
                                best_center = c_exact
        except Exception:
            pass

        # Phase 4: High-precision polish with direct objective
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    S, _, _ = calc_S_direct(pts, beta, cx, cy)
                    return S

                polish_starts = [
                    [best_center[0], best_center[1], best_beta],
                    [best_center[0], best_center[1], best_beta + 0.01],
                    [best_center[0], best_center[1], best_beta - 0.01],
                    [best_center[0] + 0.01, best_center[1], best_beta],
                    [best_center[0] - 0.01, best_center[1], best_beta],
                    [best_center[0], best_center[1] + 0.01, best_beta],
                    [best_center[0], best_center[1] - 0.01, best_beta],
                    [best_center[0] + 0.03, best_center[1] + 0.01, best_beta],
                    [best_center[0] - 0.03, best_center[1] - 0.01, best_beta],
                    [best_center[0] + 0.05, best_center[1], best_beta],
                    [best_center[0] - 0.05, best_center[1], best_beta],
                ]

                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res = minimize(objective, init, method='Nelder-Mead',
                                   options={'maxiter': 8000, 'xatol': 1e-14, 'fatol': 1e-14})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (float(res.x[0]), float(res.x[1]))
                        best_beta = ((res.x[2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
        except Exception:
            pass

        # Phase 5: Final LP verification at the best point
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                S_lp, c_lp, _ = calc_S_lp(pts, best_beta)
                if S_lp < best_S:
                    best_S = S_lp
                    best_center = c_lp
        except Exception:
            pass

        return best_S, best_beta, best_center

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
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
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        S, _, _ = calc_S_direct(best_global_pts, best_global_beta, best_global_center[0], best_global_center[1])
        best_global_S = S

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # Scale points very slightly inward to ensure strict containment
    SCALE = 1.0 - 1e-10
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))

    pts_scaled = np.array(pts_scaled)

    # Compute final enclosing size
    max_proj = 0.0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx = math.cos(ang)
        ny = math.sin(ang)
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))

    S_scaled = (max_proj + APOTHEM) / APOTHEM

    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)

    return inner, center, side, rotation
