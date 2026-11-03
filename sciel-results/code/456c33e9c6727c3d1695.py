import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
EPS = 1e-9
TIME_BUDGET = 120.0

def construct():
    if time.time() - START_TIME > TIME_BUDGET - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    candidates = []
    
    c1 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (3, 0), (3, 1)
    ]
    candidates.append(axial_to_cartesian(c1))

    c2 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    candidates.append(axial_to_cartesian(c2))

    c3 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (0, 2), (-2, 2), (2, -2)
    ]
    candidates.append(axial_to_cartesian(c3))
    
    c4 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (1, 2), (-1, 3)
    ]
    candidates.append(axial_to_cartesian(c4))
    
    c5 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_to_cartesian(c5))
    
    c6 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ]
    candidates.append(axial_to_cartesian(c6))

    def calc_S_for_beta(pts, beta):
        if time.time() - START_TIME > TIME_BUDGET - SAFETY_MARGIN:
            return float('inf'), None, None

        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        n_pts = len(pts)

        A_ub_list = []
        b_ub_list = []
        for k in range(6):
            nx, ny = normals[k]
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            A_ub_list.append(np.column_stack([-np.full(n_pts, nx), -np.full(n_pts, ny), -np.ones(n_pts)]))
            b_ub_list.append(-projs)
            A_ub_list.append(np.column_stack([np.full(n_pts, nx), np.full(n_pts, ny), -np.ones(n_pts)]))
            b_ub_list.append(projs)

        A_ub = np.vstack(A_ub_list)
        b_ub = np.concatenate(b_ub_list)
        c_obj = np.array([0.0, 0.0, 1.0])
        bounds = [(None, None), (None, None), (0, None)]

        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                cx = res.x[0]
                cy = res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def fast_exact_S(pts, cx, cy, beta):
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        centered = pts - np.array([cx, cy])
        projs = centered @ normals.T
        max_proj = np.max(np.abs(projs))
        return (max_proj + APOTHEM) / APOTHEM

    def optimize_shape(pts, current_best_S=float('inf')):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        best_pts = pts

        if time.time() - START_TIME > TIME_BUDGET - SAFETY_MARGIN:
            return float('inf'), 0.0, None, pts

        for i in range(37):
            if time.time() - START_TIME > TIME_BUDGET - SAFETY_MARGIN:
                return best_S, best_beta, best_center, best_pts
            beta = i * (math.pi / 6) / 36
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center

        if best_S >= current_best_S * 1.001 and current_best_S < float('inf'):
            return best_S, best_beta, best_center, best_pts

        try:
            if time.time() - START_TIME < TIME_BUDGET - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0], bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 300, 'xatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass

        if time.time() - START_TIME < TIME_BUDGET - SAFETY_MARGIN and best_center is not None:
            def objective(params):
                if time.time() - START_TIME > TIME_BUDGET - SAFETY_MARGIN:
                    return 1e10
                n_pts = len(pts)
                pts_new = params[:2*n_pts].reshape((n_pts, 2))
                cx = params[2*n_pts]
                cy = params[2*n_pts + 1]
                beta = params[2*n_pts + 2]
                
                penalty = 0.0
                for i in range(n_pts):
                    for j in range(i+1, n_pts):
                        dist = np.linalg.norm(pts_new[i] - pts_new[j])
                        if dist < math.sqrt(3):
                            penalty += (math.sqrt(3) - dist)**2
                if penalty > 0.1:
                    return 1e10 + penalty

                S = fast_exact_S(pts_new, cx, cy, beta)
                return S + 100 * penalty

            init_params = np.concatenate([pts.flatten(), [best_center[0], best_center[1], best_beta]])
            
            try:
                res = minimize(objective, init_params, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
                if res.fun < best_S + 1e-6:
                    n_pts = len(pts)
                    pts_new = res.x[:2*n_pts].reshape((n_pts, 2))
                    cx_new = res.x[2*n_pts]
                    cy_new = res.x[2*n_pts + 1]
                    beta_new = res.x[2*n_pts + 2]
                    
                    valid = True
                    for i in range(n_pts):
                        for j in range(i+1, n_pts):
                            if np.linalg.norm(pts_new[i] - pts_new[j]) < math.sqrt(3) - 1e-5:
                                valid = False
                                break
                        if not valid: break
                    
                    if valid:
                        S_new = fast_exact_S(pts_new, cx_new, cy_new, beta_new)
                        if S_new < best_S:
                            best_S = S_new
                            best_center = (cx_new, cy_new)
                            best_beta = beta_new
                            best_pts = pts_new
            except Exception:
                pass

        return best_S, best_beta, best_center, best_pts

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for pts in candidates:
        if time.time() - START_TIME > TIME_BUDGET - SAFETY_MARGIN:
            break
        S, beta, center, pts_opt = optimize_shape(pts, current_best_S=best_global_S)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts_opt
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = candidates[0]
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    inner = []
    for x, y in best_global_pts:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(best_global_S)
    rotation = float(angle_deg)

    return inner, center, side, rotation
