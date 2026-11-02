import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar, linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 148.0
SAFETY_MARGIN = 4.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Candidate axial coordinate configurations for 11 unit hexagons
    candidates = []

    # Ring (6) + center + 4 outer (various placements)
    ring6 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]

    # Ring + 4 axial neighbors
    candidates.append(ring6 + [(2,0),(-2,0),(0,2),(0,-2)])
    # Ring + 4 diagonal neighbors
    candidates.append(ring6 + [(1,1),(-1,2),(-2,1),(2,-1)])
    # Ring + 4 mixed
    candidates.append(ring6 + [(2,0),(-2,0),(1,1),(-1,-1)])
    candidates.append(ring6 + [(2,0),(0,2),(-2,2),(2,-2)])
    candidates.append(ring6 + [(1,0),(0,1),(-1,1),(1,-1)])  # duplicate ring, skip
    candidates.append(ring6 + [(2,0),(-2,0),(0,2),(1,-2)])
    candidates.append(ring6 + [(2,1),(-2,2),(-1,3),(1,-2)])

    # 3x3 block + 2
    b33 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2)]
    candidates.append(b33 + [(-1,1),(3,1)])
    candidates.append(b33 + [(1,3),(2,3)])
    candidates.append(b33 + [(0,3),(1,3)])
    candidates.append(b33 + [(-1,0),(3,2)])
    candidates.append(b33 + [(-1,1),(-1,2)])

    # 4x2 + 3
    b42 = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)]
    candidates.append(b42)
    candidates.append(b42 + [(1,3),(2,3)])
    candidates.append(b42 + [(-1,1),(4,1)])

    # 2x3 + 3
    b23 = [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)]
    candidates.append(b23)

    # 5+6 elongated
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)])

    # Cross
    candidates.append([(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)])

    # 2x2 + 4x1 + 1
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)])

    # 3x2 + 3x1
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])

    # 4x1 + 3x1 + 3x1
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2)])

    # 2x2 + 3x1 + 2x1
    candidates.append([(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4)])

    # 4x2 + 2x1
    candidates.append([(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)])

    # 3x2 + 4x1 + 1
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)])

    # 2x3 + 2x2
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(3,0),(3,1)])

    # Elongated 5+4
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2)])

    # 2x4 + 1
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)])

    # 4x1 + 2+2+1
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4)])

    # 3x3 + 2 top
    candidates.append(b33 + [(1,3),(2,3)])

    # Ring + 4 top
    candidates.append(ring6 + [(2,1),(1,2),(-1,2),(0,3)])

    # 2x2 + 4x1
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)])

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts, dtype=np.float64)

    def compute_normals(beta):
        return np.array([
            [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
            for k in range(6)
        ])

    def exact_S_for_beta(pts, beta):
        """LP-based exact computation of minimal enclosing hexagon side for given beta."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
        normals = compute_normals(beta)
        projs = normals @ pts.T  # (6, n)
        max_projs = projs.max(axis=1)
        min_projs = projs.min(axis=1)

        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        for k in range(6):
            A_ub.append([-normals[k, 0], -normals[k, 1], -1.0])
            b_ub.append(-max_projs[k])
            A_ub.append([normals[k, 0], normals[k, 1], -1.0])
            b_ub.append(min_projs[k])

        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]

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

    def exact_S_direct(pts, beta, cx, cy):
        """Direct evaluation without LP."""
        normals = compute_normals(beta)
        centered = pts - np.array([cx, cy])
        projs = normals @ centered.T
        max_proj = float(np.max(np.abs(projs)))
        S = (max_proj + APOTHEM) / APOTHEM
        return S

    def logsumexp_smooth(pts, beta, cx, cy, temperature=0.05):
        """Smooth surrogate via log-sum-exp."""
        normals = compute_normals(beta)
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T  # (n, 6)
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        max_val = np.max(all_projs)
        if max_val > 200:
            return 1e10
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -60, 60)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        D = temperature * log_sum
        S = (D + APOTHEM) / APOTHEM
        return S

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None

        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None

        # Coarse grid over beta in [0, pi/6] (hexagon symmetry period)
        n_coarse = 72
        for i in range(n_coarse + 1):
            beta = i * (math.pi / 6.0) / n_coarse
            S, center, D = exact_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center

        # Refine with minimize_scalar
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(
                    lambda b: exact_S_for_beta(pts, b)[0],
                    bounds=(max(0, best_beta - math.pi / 120), min(math.pi / 6, best_beta + math.pi / 120)),
                    method='bounded',
                    options={'maxiter': 300, 'xatol': 1e-14}
                )
                if res.fun < best_S:
                    best_S = float(res.fun)
                    best_beta = float(res.x)
                    _, best_center, _ = exact_S_for_beta(pts, best_beta)
        except Exception:
            pass

        # Nelder-Mead on (cx, cy, beta) with exact objective
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def obj(params):
                    cx, cy, b = params
                    b = ((b % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    return exact_S_direct(pts, b, cx, cy)

                starts = [
                    [best_center[0], best_center[1], best_beta],
                    [best_center[0], best_center[1], best_beta + 0.02],
                    [best_center[0], best_center[1], best_beta - 0.02],
                    [best_center[0] + 0.15, best_center[1], best_beta],
                    [best_center[0] - 0.15, best_center[1], best_beta],
                    [best_center[0], best_center[1] + 0.15, best_beta],
                    [best_center[0], best_center[1] - 0.15, best_beta],
                    [best_center[0] + 0.1, best_center[1] + 0.1, best_beta],
                    [best_center[0] - 0.1, best_center[1] - 0.1, best_beta],
                ]

                for init in starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    try:
                        res = minimize(obj, init, method='Nelder-Mead',
                                       options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-13})
                        if res.fun < best_S:
                            best_S = float(res.fun)
                            best_center = (float(res.x[0]), float(res.x[1]))
                            best_beta = ((float(res.x[2]) % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    except Exception:
                        pass
        except Exception:
            pass

        # Smooth surrogate refinement
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                temperatures = [0.15, 0.08, 0.04, 0.02]
                for temp in temperatures:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    def smooth_obj(params, t=temp):
                        cx, cy, b = params
                        b = ((b % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                        return logsumexp_smooth(pts, b, cx, cy, temperature=t)

                    for init in [[best_center[0], best_center[1], best_beta],
                                 [best_center[0], best_center[1], best_beta + 0.03]]:
                        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                            break
                        try:
                            res = minimize(smooth_obj, init, method='Nelder-Mead',
                                           options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12})
                            if res.fun < 1e8:
                                cx_r, cy_r, beta_r = float(res.x[0]), float(res.x[1]), float(res.x[2])
                                beta_r = ((beta_r % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                                S_exact = exact_S_direct(pts, beta_r, cx_r, cy_r)
                                if S_exact < best_S:
                                    best_S = S_exact
                                    best_center = (cx_r, cy_r)
                                    best_beta = beta_r
                        except Exception:
                            pass
        except Exception:
            pass

        # Final high-precision polish
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def obj(params):
                    cx, cy, b = params
                    b = ((b % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    return exact_S_direct(pts, b, cx, cy)

                for init in [[best_center[0], best_center[1], best_beta],
                             [best_center[0], best_center[1], best_beta + 0.005],
                             [best_center[0], best_center[1], best_beta - 0.005]]:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    try:
                        res = minimize(obj, init, method='Nelder-Mead',
                                       options={'maxiter': 20000, 'xatol': 1e-15, 'fatol': 1e-15})
                        if res.fun < best_S:
                            best_S = float(res.fun)
                            best_center = (float(res.x[0]), float(res.x[1]))
                            best_beta = ((float(res.x[2]) % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    except Exception:
                        pass
        except Exception:
            pass

        return best_S, best_beta, best_center

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

    # Fallback
    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        cx_m = float(np.mean(best_global_pts[:, 0]))
        cy_m = float(np.mean(best_global_pts[:, 1]))
        best_global_center = (cx_m, cy_m)
        best_global_beta = 0.0
        max_proj = 0.0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # Scale inner hexagons by a tiny factor to ensure strict non-intersection
    SCALE = 1.0 - 1e-10
    pts_scaled = np.array([
        [cx + (x - cx) * SCALE, cy + (y - cy) * SCALE]
        for x, y in best_global_pts
    ])

    # Compute actual enclosing hexagon side from scaled positions
    normals = compute_normals(best_global_beta)
    centered = pts_scaled - np.array([cx, cy])
    projs = normals @ centered.T
    D_scaled = float(np.max(np.abs(projs)))
    S_scaled = (D_scaled + APOTHEM) / APOTHEM

    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)

    return inner, center, side, rotation
