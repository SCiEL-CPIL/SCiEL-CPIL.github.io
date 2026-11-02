import math
import numpy as np
import time
from scipy.optimize import linprog, minimize_scalar, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 8.0
EPS = 1e-9
SQRT3 = math.sqrt(3)

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    candidates = []

    # Axial coordinate based candidates (converted to Cartesian)
    axial_configs = [
        # Ring-based: center + 6 neighbors + 4 more
        [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
         (1, 1), (-1, 2), (-2, 1), (2, -1)],
        # Ring + 4 axis extensions
        [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
         (2, 0), (-2, 0), (0, 2), (0, -2)],
        # 3x3 block + 2 extras
        [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1),
         (0, 2), (1, 2), (2, 2), (-1, 1), (3, 1)],
        # 4x3 parallelogram
        [(0, 0), (1, 0), (2, 0), (3, 0),
         (0, 1), (1, 1), (2, 1), (3, 1),
         (0, 2), (1, 2), (2, 2)],
        # 2-3-3-3 pattern
        [(0, 0), (1, 0),
         (0, 1), (1, 1), (2, 1),
         (0, 2), (1, 2), (2, 2),
         (0, 3), (1, 3), (2, 3)],
        # 3-3-2-3 pattern
        [(0, 0), (1, 0), (2, 0),
         (0, 1), (1, 1), (2, 1),
         (0, 2), (1, 2), (2, 2),
         (0, 3), (1, 3)],
        # Cross/star pattern
        [(0, 0),
         (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
         (2, 0), (-2, 0), (0, 2), (0, -2)],
        # 5-6 elongated
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
         (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)],
        # Asymmetric ring extension
        [(0, 0),
         (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
         (2, 1), (-2, 2), (-1, 3), (1, -2)],
        # 3-3-3-2 pattern
        [(0, 0), (1, 0), (2, 0),
         (0, 1), (1, 1), (2, 1),
         (0, 2), (1, 2), (2, 2),
         (1, 3), (2, 3)],
        # 2-4-4-1 pattern
        [(0, 0), (1, 0),
         (0, 1), (1, 1), (2, 1), (3, 1),
         (0, 2), (1, 2), (2, 2), (3, 2),
         (1, 3)],
        # L-shaped
        [(0, 0), (1, 0), (2, 0), (3, 0),
         (0, 1), (1, 1),
         (0, 2), (1, 2),
         (0, 3), (1, 3),
         (0, 4)],
        # Asymmetric ring
        [(0, 0),
         (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
         (2, 0), (-2, 0), (0, 2), (1, 2)],
        # Another asymmetric ring
        [(0, 0),
         (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
         (2, 0), (-2, 0), (1, 1), (-1, -1)],
        # 3-2-3-3 pattern
        [(0, 0), (1, 0), (2, 0),
         (1, 1), (2, 1),
         (0, 2), (1, 2), (2, 2),
         (-1, 1), (3, 1), (0, -1)],
        # 2-3-2-3-1 pattern
        [(0, 0), (1, 0),
         (0, 1), (1, 1), (2, 1),
         (1, 2), (2, 2),
         (0, 3), (1, 3), (2, 3), (3, 2)],
        # Star with extensions
        [(0, 0),
         (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
         (2, 1), (1, 2), (-1, 2), (0, 3)],
        # 4-3-3 pattern
        [(0, 0), (1, 0), (2, 0), (3, 0),
         (0, 1), (1, 1), (2, 1),
         (1, 2), (2, 2), (3, 2)],
        # 2-3-2-4 pattern
        [(0, 0), (1, 0),
         (0, 1), (1, 1), (2, 1),
         (1, 2), (2, 2),
         (0, 3), (1, 3), (2, 3), (3, 3)],
        # Another asymmetric
        [(0, 0),
         (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
         (2, 0), (-2, 0), (1, 1), (-1, 1)],
        # 4-5-2 pattern
        [(0, 0), (1, 0), (2, 0), (3, 0),
         (0, 1), (1, 1), (2, 1), (3, 1),
         (1, 2), (2, 2), (3, 2)],
        # 3-4-3-1 pattern
        [(0, 0), (1, 0), (2, 0),
         (0, 1), (1, 1), (2, 1), (3, 1),
         (0, 2), (1, 2), (2, 2),
         (1, 3)],
        # Symmetric cross
        [(0, 0),
         (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
         (2, 0), (-2, 0), (0, 2), (0, -2)],
        # 2-2-3-2-2 pattern
        [(0, 0), (1, 0),
         (0, 1), (1, 1),
         (0, 2), (1, 2), (2, 2),
         (0, 3), (1, 3),
         (0, 4)],
        # 3-3-3-2 pattern
        [(0, 0), (1, 0), (2, 0),
         (0, 1), (1, 1), (2, 1),
         (0, 2), (1, 2), (2, 2),
         (0, 3), (1, 3), (2, 3)],
        # 4-3-4-2 pattern
        [(0, 0), (1, 0), (2, 0), (3, 0),
         (1, 1), (2, 1), (3, 1),
         (0, 2), (1, 2), (2, 2), (3, 2),
         (1, 3), (2, 3)],
        # Asymmetric star
        [(0, 0),
         (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
         (2, 0), (-2, 0), (0, 2), (1, -2)],
        # 2-3-3-2-1 pattern
        [(0, 0), (1, 0),
         (0, 1), (1, 1), (2, 1),
         (0, 2), (1, 2),
         (0, 3), (1, 3), (2, 3),
         (1, 4), (2, 4)],
        # 3-4-3 pattern
        [(0, 0), (1, 0), (2, 0),
         (0, 1), (1, 1), (2, 1), (3, 1),
         (0, 2), (1, 2), (2, 2),
         (0, 3)],
        # 2-3-3-3 pattern (shifted)
        [(0, 0), (1, 0),
         (0, 1), (1, 1), (2, 1),
         (0, 2), (1, 2), (2, 2),
         (0, 3), (1, 3)],
        # 3-2-2-3-1 pattern
        [(0, 0), (1, 0), (2, 0),
         (0, 1), (1, 1),
         (0, 2), (1, 2), (2, 2),
         (0, 3), (1, 3),
         (0, 4), (1, 4)],
        # Another symmetric
        [(0, 0),
         (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
         (2, 0), (-2, 0), (0, 2), (0, -2)],
        # 4-3-3-1 pattern
        [(0, 0), (1, 0), (2, 0), (3, 0),
         (0, 1), (1, 1), (2, 1),
         (0, 2), (1, 2),
         (0, 3), (1, 3)],
        # 2-3-3-3 pattern
        [(0, 0), (1, 0),
         (0, 1), (1, 1), (2, 1),
         (0, 2), (1, 2), (2, 2),
         (0, 3), (1, 3), (2, 3)],
        # 3-3-3-2 pattern (different)
        [(0, 0), (1, 0), (2, 0),
         (0, 1), (1, 1), (2, 1),
         (0, 2), (1, 2), (2, 2),
         (0, 3), (1, 3), (2, 3)],
    ]

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = SQRT3 * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    # Convert all axial candidates to Cartesian
    for axial in axial_configs:
        candidates.append(axial_to_cartesian(axial))



    def calc_S_exact(pts, beta):
        """Exact LP-based computation of S for given beta."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None

        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))

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
                D = res.x[2]
                cx = res.x[0]
                cy = res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def smooth_S(pts, beta, temperature=0.05):
        """
        Smooth surrogate for S(beta) using log-sum-exp.
        Fast, suitable for gradient-based search.
        """
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf')

        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])

        center = np.mean(pts, axis=0)
        pts_centered = pts - center

        projs = pts_centered @ normals.T  # (n, 6)
        all_abs_projs = np.abs(projs).ravel()  # (n*6,)

        max_val = np.max(all_abs_projs)
        if max_val > 100:
            return float('inf')

        shifted = (all_abs_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted)))
        D_smooth = max_val + temperature * log_sum

        S_smooth = (D_smooth + APOTHEM) / APOTHEM
        return S_smooth

    def optimize_shape(pts):
        """Multi-phase optimization of beta for a given point configuration."""
        best_S = float('inf')
        best_beta = 0.0
        best_center = None

        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None

        # Phase 1: Coarse grid search with smooth surrogate
        n_coarse = 180
        coarse_best_S = float('inf')
        coarse_best_beta = 0.0
        for i in range(n_coarse + 1):
            beta = i * (math.pi / 6) / n_coarse
            S_smooth = smooth_S(pts, beta, temperature=0.1)
            if S_smooth < coarse_best_S:
                coarse_best_S = S_smooth
                coarse_best_beta = beta

        # Phase 2: Medium grid search around best
        n_medium = 200
        medium_best_S = float('inf')
        medium_best_beta = coarse_best_beta
        search_range = (math.pi / 6) / n_coarse * 2
        for i in range(n_medium + 1):
            beta = coarse_best_beta - search_range + 2 * search_range * i / n_medium
            if beta < 0:
                beta += math.pi / 6
            if beta > math.pi / 6:
                beta -= math.pi / 6
            S_smooth = smooth_S(pts, beta, temperature=0.05)
            if S_smooth < medium_best_S:
                medium_best_S = S_smooth
                medium_best_beta = beta

        # Phase 3: Scalar minimization with smooth surrogate
        try:
            for offset in [0.0, 0.005, -0.005, 0.01, -0.01]:
                lo = max(0.0, medium_best_beta + offset - 0.02)
                hi = min(math.pi / 6, medium_best_beta + offset + 0.02)
                if lo >= hi:
                    continue
                res = minimize_scalar(lambda b: smooth_S(pts, b, temperature=0.03),
                                     bounds=(lo, hi), method='bounded',
                                     options={'maxiter': 500, 'xatol': 1e-14})
                if res.fun < medium_best_S:
                    medium_best_S = res.fun
                    medium_best_beta = res.x
        except Exception:
            pass

        # Phase 4: Verify with exact LP at smooth optimum
        S_exact, center_exact, D_exact = calc_S_exact(pts, medium_best_beta)
        if S_exact < best_S:
            best_S = S_exact
            best_beta = medium_best_beta
            best_center = center_exact

        # Phase 5: Fine grid search with exact LP
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            n_fine = 100
            fine_range = 0.01
            for i in range(n_fine + 1):
                beta = best_beta - fine_range + 2 * fine_range * i / n_fine
                if beta < 0:
                    beta += math.pi / 6
                if beta > math.pi / 6:
                    beta -= math.pi / 6
                S, center, D = calc_S_exact(pts, beta)
                if S < best_S:
                    best_S = S
                    best_beta = beta
                    best_center = center

        # Phase 6: Very fine polish with exact LP
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            n_very_fine = 200
            vf_range = 0.001
            for i in range(n_very_fine + 1):
                beta = best_beta - vf_range + 2 * vf_range * i / n_very_fine
                if beta < 0:
                    beta += math.pi / 6
                if beta > math.pi / 6:
                    beta -= math.pi / 6
                S, center, D = calc_S_exact(pts, beta)
                if S < best_S:
                    best_S = S
                    best_beta = beta
                    best_center = center

        # Phase 7: Ultra-fine final polish
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            n_ultra = 100
            uf_range = 1e-4
            for i in range(n_ultra + 1):
                beta = best_beta - uf_range + 2 * uf_range * i / n_ultra
                if beta < 0:
                    beta += math.pi / 6
                if beta > math.pi / 6:
                    beta -= math.pi / 6
                S, center, D = calc_S_exact(pts, beta)
                if S < best_S:
                    best_S = S
                    best_beta = beta
                    best_center = center

        return best_S, best_beta, best_center

    def local_refinement(pts, beta, center, S, n_iterations=30):
        """
        Try to improve the packing by slightly perturbing individual hexagon positions.
        This allows for non-grid-optimal configurations.
        """
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return S, beta, center, pts

        best_S = S
        best_pts = pts.copy()
        best_center = center

        cx, cy = center
        for iteration in range(n_iterations):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break

            improved = False
            for i in range(len(pts)):
                # Try small perturbations in x and y for hexagon i
                for dx in [-0.05, -0.02, 0.0, 0.02, 0.05]:
                    for dy in [-0.05, -0.02, 0.0, 0.02, 0.05]:
                        if dx == 0 and dy == 0:
                            continue
                        new_pts = best_pts.copy()
                        new_pts[i] = [best_pts[i][0] + dx, best_pts[i][1] + dy]
                        
                        S_test, center_test, D_test = calc_S_exact(new_pts, beta)
                        if S_test < best_S - 1e-10:
                            best_S = S_test
                            best_pts = new_pts
                            best_center = center_test
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

            if not improved:
                break

        return best_S, beta, best_center, best_pts

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    # Evaluate all candidates
    for idx, pts in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        # Quick estimate to skip obviously bad candidates
        x_range = np.max(pts[:, 0]) - np.min(pts[:, 0])
        y_range = np.max(pts[:, 1]) - np.min(pts[:, 1])
        quick_S = max(x_range, y_range) / (2 * APOTHEM)
        
        if quick_S > best_global_S + 0.5 and best_global_S < float('inf'):
            continue

        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    # Local refinement is disabled to ensure validity (no overlap checks in perturbations)
    # The axial grid candidates are structurally non-overlapping.

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = candidates[0]
        best_global_center = (float(np.mean(best_global_pts[:, 0])),
                              float(np.mean(best_global_pts[:, 1])))
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

    SCALE = 1.0 + 1e-12
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))

    pts_scaled = np.array(pts_scaled)

    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))

    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM

    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)

    return inner, center, side, rotation
