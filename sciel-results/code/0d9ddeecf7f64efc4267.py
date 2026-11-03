import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar, linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 8.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Candidate configurations in axial coordinates
    candidates = []

    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    candidates.append(axial_1)

    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_2)

    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    candidates.append(axial_3)

    axial_4 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_4)

    axial_5 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_5)

    axial_6 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_6)

    axial_7 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_7)

    axial_8 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    candidates.append(axial_8)

    axial_9 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ]
    candidates.append(axial_9)

    axial_10 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_10)

    axial_11 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3)
    ]
    candidates.append(axial_11)

    axial_12 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_12)

    axial_13 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 2)
    ]
    candidates.append(axial_13)

    axial_14 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ]
    candidates.append(axial_14)

    axial_15 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1), (0, -1)
    ]
    candidates.append(axial_15)

    axial_16 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_16)

    axial_17 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3)
    ]
    candidates.append(axial_17)

    axial_18 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_18)

    axial_19 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_19)

    axial_20 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, 1)
    ]
    candidates.append(axial_20)

    axial_21 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_21)

    axial_22 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ]
    candidates.append(axial_22)

    axial_23 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_23)

    axial_24 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_24)

    axial_25 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_25)

    axial_26 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_26)

    axial_27 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, -2)
    ]
    candidates.append(axial_27)

    axial_28 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4), (2, 4)
    ]
    candidates.append(axial_28)

    axial_29 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, -1), (2, -1)
    ]
    candidates.append(axial_29)

    axial_30 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3),
        (1, -1)
    ]
    candidates.append(axial_30)

    axial_31 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3),
        (1, -1), (2, -1)
    ]
    candidates.append(axial_31)

    axial_32 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, -2)
    ]
    candidates.append(axial_32)

    axial_33 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (0, 3), (-1, 2)
    ]
    candidates.append(axial_33)

    axial_34 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4)
    ]
    candidates.append(axial_34)

    axial_35 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4), (1, 4)
    ]
    candidates.append(axial_35)

    axial_36 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4)
    ]
    candidates.append(axial_36)

    axial_37 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4), (2, 4)
    ]
    candidates.append(axial_37)

    axial_38 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_38)

    axial_39 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3),
        (0, 4)
    ]
    candidates.append(axial_39)

    axial_40 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4), (1, 4)
    ]
    candidates.append(axial_40)

    # Additional diverse candidates for better coverage
    axial_41 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_41)

    axial_42 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_42)

    axial_43 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 1)
    ]
    candidates.append(axial_43)

    axial_44 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (1, -1)
    ]
    candidates.append(axial_44)

    axial_45 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3),
        (0, 4)
    ]
    candidates.append(axial_45)

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def calc_S_exact(pts, beta, cx=None, cy=None):
        """Compute exact S for given beta and optionally fixed center."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None

        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])

        if cx is None or cy is None:
            # Use LP to find optimal center
            n_pts = len(pts)
            projs = pts @ normals.T  # (n_pts, 6)

            A_ub = np.zeros((2 * 6 * n_pts, 3))
            b_ub = np.zeros(2 * 6 * n_pts)

            for k in range(6):
                nx, ny = normals[k]
                base = k * 2 * n_pts
                A_ub[base:base+n_pts, 0] = nx
                A_ub[base:base+n_pts, 1] = ny
                A_ub[base:base+n_pts, 2] = -1.0
                b_ub[base:base+n_pts] = projs[:, k]

                A_ub[base+n_pts:base+2*n_pts, 0] = -nx
                A_ub[base+n_pts:base+2*n_pts, 1] = -ny
                A_ub[base+n_pts:base+2*n_pts, 2] = -1.0
                b_ub[base+n_pts:base+2*n_pts] = -projs[:, k]

            c_obj = np.array([0.0, 0.0, 1.0])
            bounds = [(None, None), (None, None), (0, None)]

            try:
                res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if res.status == 0:
                    cx = float(res.x[0])
                    cy = float(res.x[1])
                    D = float(res.x[2])
                else:
                    cx0 = float(np.mean(pts[:, 0]))
                    cy0 = float(np.mean(pts[:, 1]))
                    pts_centered = pts - np.array([cx0, cy0])
                    D = float(np.max(np.abs(pts_centered @ normals.T)))
            except Exception:
                cx0 = float(np.mean(pts[:, 0]))
                cy0 = float(np.mean(pts[:, 1]))
                pts_centered = pts - np.array([cx0, cy0])
                D = float(np.max(np.abs(pts_centered @ normals.T)))
        else:
            pts_centered = pts - np.array([cx, cy])
            projs = np.abs(pts_centered @ normals.T)
            D = float(np.max(projs))

        S = (D + APOTHEM) / APOTHEM
        return S, (cx, cy), D

    def smooth_calc_S(pts, beta, cx, cy, temperature=0.05):
        """Smooth surrogate using log-sum-exp for the max projection."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])

        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T  # (n_pts, 6)

        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        max_val = np.max(all_projs)
        if max_val > 1000:
            return float('inf')

        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted)))
        D = temperature * (log_sum + max_val / temperature)
        S = (D + APOTHEM) / APOTHEM
        return S

    def newton_polish_center(pts, beta, cx0, cy0, max_iter=300, tol=1e-14):
        """High-precision Newton-style polish for center at fixed rotation."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])

        cx, cy = cx0, cy0
        for it in range(max_iter):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            pts_centered = pts - np.array([cx, cy])
            projs = pts_centered @ normals.T
            abs_projs = np.abs(projs)
            max_proj = np.max(abs_projs)

            active_mask = abs_projs >= max_proj - 1e-12
            active_pts_idx, active_norm_idx = np.where(active_mask)

            if len(active_pts_idx) < 2:
                break

            s = np.sign(projs[active_pts_idx, active_norm_idx])
            A = np.zeros((len(s), 3))
            b = np.zeros(len(s))
            for j in range(len(s)):
                i = active_pts_idx[j]
                k = active_norm_idx[j]
                A[j, 0] = s[j] * normals[k, 0]
                A[j, 1] = s[j] * normals[k, 1]
                A[j, 2] = 1.0
                b[j] = s[j] * (pts[i, 0] * normals[k, 0] + pts[i, 1] * normals[k, 1])

            try:
                sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                if rank >= 2:
                    new_cx, new_cy, new_D = sol
                    pts_c_new = pts - np.array([new_cx, new_cy])
                    projs_new = np.abs(pts_c_new @ normals.T)
                    max_new = np.max(projs_new)
                    if max_new > max_proj * (1 + 1e-10):
                        new_cx = cx + 0.5 * (new_cx - cx)
                        new_cy = cy + 0.5 * (new_cy - cy)
                    cx, cy = new_cx, new_cy
                    if abs(new_D - max_proj) < tol:
                        break
                else:
                    break
            except Exception:
                break

        return cx, cy

    def optimize_shape(pts):
        """Optimize S for a given cluster shape using multi-stage approach."""
        best_S = float('inf')
        best_beta = 0.0
        best_center = None

        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None

        # Stage 1: Coarse grid search over beta in [0, pi/6)
        n_coarse = 48
        for i in range(n_coarse):
            beta = i * (math.pi / 6) / n_coarse
            S, center, D = calc_S_exact(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center

        # Stage 2: Bounded scalar refinement
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: calc_S_exact(pts, b)[0],
                                     bounds=(max(0, best_beta - 0.05), min(math.pi/6, best_beta + 0.05)),
                                     method='bounded', options={'maxiter': 200, 'xatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_exact(pts, best_beta)
        except Exception:
            pass

        # Stage 3: Smooth surrogate optimization with multiple temperatures
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                temperatures = [0.2, 0.1, 0.05, 0.02, 0.01]
                start_betas = [best_beta, best_beta + 0.03, best_beta - 0.03, 0.0, math.pi/12, math.pi/6 - 0.01]

                for temp in temperatures:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    for sb in start_betas:
                        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                            break

                        def smooth_obj(params, t=temp):
                            cx, cy, beta = params
                            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                            return smooth_calc_S(pts, beta, cx, cy, temperature=t)

                        init = [best_center[0], best_center[1], sb]
                        try:
                            res = minimize(smooth_obj, init, method='Nelder-Mead',
                                         options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-10})
                            if res.fun < 1e10:
                                cx_r, cy_r, beta_r = res.x
                                beta_r = ((beta_r % (math.pi/6)) + math.pi/6) % (math.pi/6)
                                S_exact, c_exact, _ = calc_S_exact(pts, beta_r)
                                if S_exact < best_S:
                                    best_S = S_exact
                                    best_beta = beta_r
                                    best_center = c_exact
                        except Exception:
                            pass
        except Exception:
            pass

        # Stage 4: Exact Nelder-Mead polish on the true objective
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
                    normals = np.column_stack([np.cos(angles), np.sin(angles)])
                    pts_centered = pts - np.array([cx, cy])
                    projs = np.abs(pts_centered @ normals.T)
                    max_proj = float(np.max(projs))
                    S = (max_proj + APOTHEM) / APOTHEM
                    return S

                polish_starts = [
                    [best_center[0], best_center[1], best_beta],
                    [best_center[0], best_center[1], best_beta + 0.015],
                    [best_center[0], best_center[1], best_beta - 0.015],
                    [best_center[0] + 0.015, best_center[1], best_beta],
                    [best_center[0] - 0.015, best_center[1], best_beta],
                    [best_center[0], best_center[1] + 0.015, best_beta],
                    [best_center[0], best_center[1] - 0.015, best_beta],
                ]

                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    try:
                        res = minimize(objective, init, method='Nelder-Mead',
                                     options={'maxiter': 8000, 'xatol': 1e-14, 'fatol': 1e-14})
                        if res.fun < best_S:
                            best_S = res.fun
                            best_center = (float(res.x[0]), float(res.x[1]))
                            best_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    except Exception:
                        pass
        except Exception:
            pass

        # Stage 5: High-precision Newton polish for center at fixed rotation
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                cx_p, cy_p = newton_polish_center(pts, best_beta, best_center[0], best_center[1])
                S_f, (cx_f, cy_f), D_f = calc_S_exact(pts, best_beta, cx_p, cy_p)
                if S_f < best_S:
                    best_S = S_f
                    best_center = (cx_f, cy_f)
        except Exception:
            pass

        return best_S, best_beta, best_center

    # Main loop: evaluate all candidates
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for idx, axial in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    # Fallback if no valid result
    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        angles = math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        pts_centered = best_global_pts - np.array(best_global_center)
        max_proj = float(np.max(np.abs(pts_centered @ normals.T)))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # Final verification and safety margin
    pts_scaled = best_global_pts.copy()
    max_proj = 0
    angles = best_global_beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    pts_centered = pts_scaled - np.array([cx, cy])
    projs = np.abs(pts_centered @ normals.T)
    D_final = float(np.max(projs))
    S_final = (D_final + APOTHEM) / APOTHEM

    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(angle_deg)

    return inner, center, side, rotation
