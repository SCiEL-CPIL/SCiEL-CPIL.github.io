import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

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

    # Center + 6 neighbors + 4 more (various patterns)
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
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 1), (-2, 2), (1, -2), (-1, -2)
    ]
    candidates.append(axial_29)

    axial_30 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (1, 4)
    ]
    candidates.append(axial_30)

    axial_31 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3),
        (2, 3), (3, 2)
    ]
    candidates.append(axial_31)

    axial_32 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (1, -2), (-1, -2)
    ]
    candidates.append(axial_32)

    # Additional corner-filling patterns
    axial_33 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (1, -2)
    ]
    candidates.append(axial_33)

    axial_34 = [
        (0, 0), (1, 0), (-1, 0),
        (0, 1), (1, 1), (-1, 1),
        (0, 2), (1, 2), (-1, 2),
        (0, -1), (1, -1)
    ]
    candidates.append(axial_34)

    axial_35 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_35)

    axial_36 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 1), (-2, 2), (0, 2)
    ]
    candidates.append(axial_36)

    axial_37 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3),
        (0, 4)
    ]
    candidates.append(axial_37)

    axial_38 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (0, 2)
    ]
    candidates.append(axial_38)

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

    def calc_S_for_beta(pts, beta, center=None):
        """LP over center for fixed beta. Returns (S, center, D)."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, float('inf')

        normals = get_normals(beta)
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []

        for nx, ny in normals:
            for px, py in pts:
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny)

        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]

        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = float(res.x[2])
                cx, cy = float(res.x[0]), float(res.x[1])
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, float('inf')
        except Exception:
            return float('inf'), None, float('inf')

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

    def exact_objective_3d(pts, params):
        """Exact objective over (cx, cy, beta) — for final polish."""
        cx, cy, beta = params
        beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
        normals = get_normals(beta)
        D = compute_D(pts, (cx, cy), normals)
        return (D + APOTHEM) / APOTHEM

    def optimize_shape(pts):
        """Optimize (center, rotation) for a fixed point set."""
        best_S = float('inf')
        best_beta = 0.0
        best_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))

        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, best_center

        # --- Phase 1: Coarse grid over beta ---
        n_coarse = 73
        for i in range(n_coarse):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            beta = i * (math.pi / 6) / (n_coarse - 1)
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                if center is not None:
                    best_center = center

        # --- Phase 2: Bounded scalar refine around best beta ---
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                lo = max(0.0, best_beta - math.pi / 72)
                hi = min(math.pi / 6, best_beta + math.pi / 72)
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0],
                                      bounds=(lo, hi), method='bounded',
                                      options={'maxiter': 500, 'xatol': 1e-14})
                if res.fun < best_S:
                    best_S = float(res.fun)
                    best_beta = float(res.x)
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass

        # --- Phase 3: Smooth-surrogate Nelder-Mead with annealing ---
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                temperatures = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
                start_betas = [best_beta, best_beta + 0.03, best_beta - 0.03,
                               0.0, math.pi / 12, math.pi / 6,
                               best_beta + math.pi / 24, best_beta - math.pi / 24]

                for temp in temperatures:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    for sb in start_betas:
                        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                            break

                        def smooth_obj(params, t=temp):
                            cx, cy, beta = params
                            beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                            return smooth_calc_S(pts, beta, (cx, cy), t)

                        init = [best_center[0], best_center[1], sb]
                        try:
                            res = minimize(smooth_obj, init, method='Nelder-Mead',
                                           options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
                            if res.fun < 1e10:
                                cx_r, cy_r, beta_r = res.x
                                beta_r = ((beta_r % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                                S_exact, c_exact, _ = calc_S_for_beta(pts, beta_r)
                                if S_exact < best_S:
                                    best_S = S_exact
                                    best_beta = beta_r
                                    if c_exact is not None:
                                        best_center = c_exact
                        except Exception:
                            pass
        except Exception:
            pass

        # --- Phase 4: Exact high-precision polish ---
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                cx0, cy0 = best_center
                polish_starts = [
                    [cx0, cy0, best_beta],
                    [cx0, cy0, best_beta + 0.01],
                    [cx0, cy0, best_beta - 0.01],
                    [cx0 + 0.01, cy0, best_beta],
                    [cx0, cy0 + 0.01, best_beta],
                    [cx0 - 0.01, cy0, best_beta],
                    [cx0, cy0 - 0.01, best_beta],
                    [cx0 + 0.005, cy0 + 0.005, best_beta],
                    [cx0 - 0.005, cy0 - 0.005, best_beta],
                ]

                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    try:
                        res = minimize(exact_objective_3d, init, method='Nelder-Mead',
                                       options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-14})
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
                                        options={'maxiter': 1000, 'ftol': 1e-15})
                        if res2.fun < best_S:
                            best_S = float(res2.fun)
                            best_center = (float(res2.x[0]), float(res2.x[1]))
                            best_beta = ((float(res2.x[2]) % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    except Exception:
                        pass
        except Exception:
            pass

        # --- Phase 5: Tight LP-based beta refinement ---
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                n_fine = 200
                span = math.pi / 6 / 40
                for i in range(n_fine):
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    b = best_beta + (i / (n_fine - 1) - 0.5) * 2 * span
                    b = ((b % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    S, center, D = calc_S_for_beta(pts, b)
                    if S < best_S:
                        best_S = S
                        best_beta = b
                        if center is not None:
                            best_center = center

                if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                    lo = max(0.0, best_beta - span / 2)
                    hi = min(math.pi / 6, best_beta + span / 2)
                    try:
                        res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0],
                                              bounds=(lo, hi), method='bounded',
                                              options={'maxiter': 2000, 'xatol': 1e-16})
                        if res.fun < best_S:
                            best_S = float(res.fun)
                            best_beta = float(res.x)
                            _, best_center, _ = calc_S_for_beta(pts, best_beta)
                    except Exception:
                        pass
        except Exception:
            pass

        return best_S, best_beta, best_center

    # --- Main loop over candidates ---
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

    # --- Final: recompute exact S with tiny scale for safety ---
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
