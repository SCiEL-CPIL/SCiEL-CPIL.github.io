import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar, linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # --- Candidate point sets (axial coordinates) ---
    candidates = []

    # Hexagonal cluster: center + 6 neighbors + 4 more
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    candidates.append(axial_1)

    # Star: center + 6 neighbors + 4 opposite
    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_2)

    # 3x3 block + 2 extras
    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    candidates.append(axial_3)

    # 4x3 block + 1 extra
    axial_4 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_4)

    # 3x4 block + 1 extra
    axial_5 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_5)

    # Cross-like
    axial_7 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_7)

    # Center + 6 + 4 asymmetric
    axial_9 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ]
    candidates.append(axial_9)

    # 3x3 + 2 top
    axial_10 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_10)

    # 2+4+4+1
    axial_11 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3)
    ]
    candidates.append(axial_11)

    # Center + 6 + 4 (variant)
    axial_13 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 2)
    ]
    candidates.append(axial_13)

    # Center + 6 + 4 (variant)
    axial_14 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ]
    candidates.append(axial_14)

    # 3x3 with extras
    axial_15 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1), (0, -1)
    ]
    candidates.append(axial_15)

    # Staircase
    axial_16 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_16)

    # Center + 6 + 4 (variant)
    axial_17 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3)
    ]
    candidates.append(axial_17)

    # 4+3+3
    axial_18 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_18)

    # 2+3+2+3+1
    axial_19 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_19)

    # Center + 6 + 4 (variant)
    axial_20 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, 1)
    ]
    candidates.append(axial_20)

    # 4+4+3
    axial_21 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_21)

    # 3+4+3+1
    axial_22 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ]
    candidates.append(axial_22)

    # Center + 6 + 4 (variant)
    axial_23 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_23)

    # 3+2+3+3
    axial_25 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_25)

    # 4+3+4+2
    axial_26 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_26)

    # Center + 6 + 4 (variant)
    axial_27 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, -2)
    ]
    candidates.append(axial_27)

    # More diverse configurations
    axial_29 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2), (4, 2)
    ]
    candidates.append(axial_29)

    axial_30 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 1), (-2, 1), (0, 2), (1, -2)
    ]
    candidates.append(axial_30)

    axial_31 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_31)

    axial_32 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2), (4, 2)
    ]
    candidates.append(axial_32)

    axial_33 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
        (0, 2), (1, 2)
    ]
    candidates.append(axial_33)

    axial_34 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 2), (-1, 2)
    ]
    candidates.append(axial_34)

    axial_35 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_35)

    # --- Geometry helpers ---

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def get_normals(beta):
        """Return 6 unit normals of the containing hexagon for rotation beta."""
        return np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])

    def compute_D(pts, center, normals):
        """Exact: D = max over points and normals of |(p - c) . n|."""
        centered = pts - np.array(center)
        projs = centered @ normals.T  # (n_pts, 6)
        return float(np.max(np.abs(projs)))

    def calc_S_for_beta(pts, beta):
        """
        Calculate minimal side length S for a fixed rotation beta.
        Uses exact LP to find optimal center minimizing D.
        """
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None

        normals = get_normals(beta)
        n_pts = len(pts)
        # Variables: cx, cy, t (minimize t s.t. for all p, n: (p-c).n <= t and -(p-c).n <= t)
        # i.e., p.n - c.n <= t  and  -p.n + c.n <= t
        # A @ [cx, cy, t] <= b
        # For each point i and normal k:
        #   pts[i].n[k] - (cx*nx + cy*ny) <= t  =>  -nx*cx - ny*cy - t <= -pts[i].n[k]
        #   -pts[i].n[k] + (cx*nx + cy*ny) <= t  =>  nx*cx + ny*cy - t <= pts[i].n[k]
        A_rows = []
        b_rows = []
        for i in range(n_pts):
            for k in range(6):
                nx, ny = normals[k]
                proj = pts[i, 0] * nx + pts[i, 1] * ny
                # -nx*cx - ny*cy - t <= -proj
                A_rows.append([-nx, -ny, -1.0])
                b_rows.append(-proj)
                # nx*cx + ny*cy - t <= proj
                A_rows.append([nx, ny, -1.0])
                b_rows.append(proj)
        A = np.array(A_rows)
        b = np.array(b_rows)
        c_obj = np.array([0.0, 0.0, 1.0])
        try:
            res = linprog(c_obj, A_ub=A, b_ub=b, bounds=[(None, None), (None, None), (0, None)])
            if res.status == 0:
                D_opt = float(res.fun)
                c_opt = (float(res.x[0]), float(res.x[1]))
            else:
                # Fallback
                c0 = np.mean(pts, axis=0)
                centered = pts - c0
                projs = centered @ normals.T
                D_opt = float(np.max(np.abs(projs)))
                c_opt = (float(c0[0]), float(c0[1]))
        except Exception:
            c0 = np.mean(pts, axis=0)
            centered = pts - c0
            projs = centered @ normals.T
            D_opt = float(np.max(np.abs(projs)))
            c_opt = (float(c0[0]), float(c0[1]))

        S = (D_opt + APOTHEM) / APOTHEM
        return S, c_opt

    def smooth_D(pts, center, normals, temperature):
        """Smooth surrogate for D via log-sum-exp soft-max over |projections|."""
        centered = pts - np.array(center)
        projs = centered @ normals.T  # (n_pts, 6)
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        max_val = float(np.max(all_projs))
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -60.0, 60.0)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        return temperature * log_sum

    def smooth_calc_S(pts, beta, center, temperature):
        normals = get_normals(beta)
        D = smooth_D(pts, center, normals, temperature)
        S = (D + APOTHEM) / APOTHEM
        return S

    def epigraph_objective(pts, params, temperature):
        """
        Epigraph formulation: minimize t + penalty
        s.t. smooth constraints: smooth_D <= t
        Uses a smooth penalty for the constraint.
        """
        cx, cy, t = params
        beta = ((params[3] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6) if len(params) > 3 else 0.0
        normals = get_normals(beta)
        D_smooth = smooth_D(pts, (cx, cy), normals, temperature)
        # Penalty for D_smooth > t
        violation = max(0.0, D_smooth - t)
        penalty = 100.0 * violation + 10000.0 * violation * violation
        return t + penalty

    def exact_objective_3d(pts, params):
        """Exact objective over (cx, cy, beta) — for final polish."""
        cx, cy, beta = params
        beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
        normals = get_normals(beta)
        D = compute_D(pts, (cx, cy), normals)
        return (D + APOTHEM) / APOTHEM

    def optimize_shape(pts):
        """Optimize (center, rotation) for a fixed point set.
        Strategy:
          1. Coarse grid search over beta with exact LP center solve.
          2. Bounded scalar refine around best beta.
          3. Smooth epigraph Nelder-Mead in (cx, cy, t, beta) with annealing.
          4. Exact high-precision polish from multiple starts.
        """
        best_S = float('inf')
        best_beta = 0.0
        best_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))

        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, best_center

        # --- Phase 1: Coarse grid over beta (exact LP per beta) ---
        n_coarse = 121
        for i in range(n_coarse):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            beta = i * (math.pi / 6) / (n_coarse - 1)
            S, center = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                if center is not None:
                    best_center = center

        # --- Phase 2: Bounded scalar refine around best beta (multiple windows) ---
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                windows = [
                    (max(0.0, best_beta - math.pi / 30), min(math.pi / 6, best_beta + math.pi / 30)),
                    (max(0.0, best_beta - math.pi / 60), min(math.pi / 6, best_beta + math.pi / 60)),
                    (max(0.0, best_beta - math.pi / 120), min(math.pi / 6, best_beta + math.pi / 120)),
                    (max(0.0, best_beta - math.pi / 240), min(math.pi / 6, best_beta + math.pi / 240)),
                ]
                for lo, hi in windows:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    if hi - lo < 1e-15:
                        continue
                    try:
                        res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0],
                                              bounds=(lo, hi), method='bounded',
                                              options={'maxiter': 800, 'xatol': 1e-15})
                        if res.fun < best_S:
                            best_S = float(res.fun)
                            best_beta = float(res.x)
                            _, best_center = calc_S_for_beta(pts, best_beta)
                    except Exception:
                        pass
        except Exception:
            pass

        # --- Phase 3: Smooth epigraph Nelder-Mead with annealing ---
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                temperatures = [0.15, 0.08, 0.04, 0.02, 0.01, 0.005, 0.002]
                start_betas = [best_beta, best_beta + 0.02, best_beta - 0.02,
                               0.0, math.pi / 12, math.pi / 6]

                for temp in temperatures:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    for sb in start_betas:
                        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                            break

                        def smooth_obj_4d(params, t=temp):
                            cx, cy, t_val, beta = params
                            beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                            normals = get_normals(beta)
                            D_smooth = smooth_D(pts, (cx, cy), normals, t)
                            violation = max(0.0, D_smooth - t_val)
                            penalty = 50.0 * violation + 5000.0 * violation * violation
                            return t_val + penalty

                        init = [best_center[0], best_center[1], best_S * APOTHEM - APOTHEM, sb]
                        try:
                            res = minimize(smooth_obj_4d, init, method='Nelder-Mead',
                                           options={'maxiter': 600, 'xatol': 1e-13, 'fatol': 1e-13})
                            if res.fun < 1e10:
                                cx_r, cy_r, t_r, beta_r = res.x
                                beta_r = ((beta_r % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                                S_exact, c_exact = calc_S_for_beta(pts, beta_r)
                                if S_exact < best_S:
                                    best_S = S_exact
                                    best_beta = beta_r
                                    if c_exact is not None:
                                        best_center = c_exact
                        except Exception:
                            pass
        except Exception:
            pass

        # --- Phase 4: High-precision polish ---
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                cx0, cy0 = best_center

                # 4a: Very fine scalar search on beta with LP center solve
                for window in [math.pi / 400, math.pi / 800, math.pi / 1600, math.pi / 3200]:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    lo = max(0.0, best_beta - window)
                    hi = min(math.pi / 6, best_beta + window)
                    if hi - lo < 1e-15:
                        continue
                    try:
                        res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0],
                                              bounds=(lo, hi), method='bounded',
                                              options={'maxiter': 1200, 'xatol': 1e-16})
                        if res.fun < best_S:
                            best_S = float(res.fun)
                            best_beta = float(res.x)
                            _, best_center = calc_S_for_beta(pts, best_beta)
                    except Exception:
                        pass

                # 4b: Nelder-Mead + SLSQP polish in 3D
                polish_starts = [
                    [cx0, cy0, best_beta],
                    [cx0, cy0, best_beta + 0.003],
                    [cx0, cy0, best_beta - 0.003],
                    [cx0 + 0.003, cy0, best_beta],
                    [cx0, cy0 + 0.003, best_beta],
                    [cx0 - 0.003, cy0, best_beta],
                    [cx0, cy0 - 0.003, best_beta],
                    [cx0 + 0.002, cy0 + 0.002, best_beta],
                    [cx0 - 0.002, cy0 - 0.002, best_beta],
                ]

                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    try:
                        res = minimize(exact_objective_3d, init, method='Nelder-Mead',
                                       options={'maxiter': 4000, 'xatol': 1e-15, 'fatol': 1e-15})
                        if res.fun < best_S:
                            best_S = float(res.fun)
                            best_center = (float(res.x[0]), float(res.x[1]))
                            best_beta = ((float(res.x[2]) % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    except Exception:
                        pass

                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        continue
                    try:
                        res2 = minimize(exact_objective_3d,
                                        [best_center[0], best_center[1], best_beta],
                                        method='SLSQP',
                                        options={'maxiter': 2000, 'ftol': 1e-18})
                        if res2.fun < best_S:
                            best_S = float(res2.fun)
                            best_center = (float(res2.x[0]), float(res2.x[1]))
                            best_beta = ((float(res2.x[2]) % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    except Exception:
                        pass

                # 4c: Final LP center re-solve at the best beta
                if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                    S_final, c_final = calc_S_for_beta(pts, best_beta)
                    if c_final is not None and S_final < best_S:
                        best_S = S_final
                        best_center = c_final
        except Exception:
            pass

        return best_S, best_beta, best_center

    # --- Main loop over candidates ---
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    # Sort candidates by a quick heuristic: bounding box area in axial space
    def axial_bbox_area(axial_list):
        qs = [q for q, r in axial_list]
        rs = [r for r, c in axial_list]
        return (max(qs) - min(qs) + 1) * (max(rs) - min(rs) + 1)

    sorted_candidates = sorted(candidates, key=axial_bbox_area)

    for idx, axial in enumerate(sorted_candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    # --- Fallback ---
    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(best_global_pts[:, 0])),
                              float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        normals = get_normals(best_global_beta)
        max_proj = 0.0
        for nx, ny in normals:
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # --- Final: recompute exact S with the chosen (center, beta), add tiny scale for safety ---
    SCALE = 1.0 + 1e-12
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    pts_scaled = np.array(pts_scaled)

    normals = get_normals(best_global_beta)
    max_proj = 0.0
    for nx, ny in normals:
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))

    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM

    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)

    return inner, center, side, rotation
