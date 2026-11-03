import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar, linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_BUDGET = 145.0
SAFETY_MARGIN = 3.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_BUDGET - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # ---- Build candidate axial coordinate sets ----
    base_sets = [
        # Set 1: center + ring + 4 extra
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)],
        # Set 2: center + ring + 4 axial extras
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)],
        # Set 3: 3x3 block + 2
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)],
        # Set 4: 4x3 rectangle
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)],
        # Set 5: 2x4 + 3
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)],
        # Set 6: 3x4 minus 1
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)],
        # Set 7: star with 8-point ring
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)],
        # Set 8: 5+6 two-row
        [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)],
        # Set 9: center + ring + 4 diagonal
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)],
        # Set 10: 3x3 + 2 top
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)],
        # Set 11: 2x5 + 1
        [(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)],
        # Set 12: 4x3 L-shape
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4)],
        # Set 13: center + ring + 2 more
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)],
        # Set 14: center + ring + 2 diagonal
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)],
        # Set 15: 3x3 + 3 extras
        [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1),(0,-1)],
        # Set 16: 2x3 + 2x2 + 1
        [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)],
        # Set 17: center + ring + 4 more spread
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)],
        # Set 18: 4x3 + 1
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2),(0,2)],
        # Set 19: 2x4 + 2x2
        [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)],
        # Set 20: center + ring + 2 opposite
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,1)],
        # Set 21: 4x3 + 3
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)],
        # Set 22: 3x4 + 1
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)],
        # Set 23: center + ring + 4 (alt)
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)],
        # Set 24: 2x5 + 1 (tall)
        [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4),(1,4)],
        # Set 25: 3x4 full
        [(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)],
        # Set 26: 4x3 + 3 (wide)
        [(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)],
        # Set 27: center + ring + 4 (alt2)
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,-2)],
        # Set 28: 2x4 + 2x2 + 1
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(2,3),(1,4),(2,4)],
        # Set 29: 5x2 + 1
        [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(1,2)],
        # Set 30: 3x3 + 2 (compact)
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,0),(3,0)],
        # Set 31: 2x3 + 2x2 + 1 (compact)
        [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,2),(1,3),(2,3),(0,3)],
        # Set 32: 4x2 + 3
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)],
        # Set 33: 3x3 + 2 (alt)
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(2,3)],
        # Set 34: 2x4 + 3
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(3,1)],
        # Set 35: 5x2 + 1 (alt)
        [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(2,2)],
        # Set 36: 3x3 + 2 (alt2)
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(1,-1)],
        # Set 37: 4x3 (compact)
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)],
        # Set 38: 2x5 (tall)
        [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4),(1,4),(2,2)],
        # Set 39: 3x4 (compact)
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(2,3)],
        # Set 40: 4x3 (alt)
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(2,2),(3,2)],
        # Additional sets for better coverage
        # Set 41: 3x3 + 2 (corner)
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(3,2)],
        # Set 42: 4x3 (corner)
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(3,2)],
        # Set 43: 2x4 + 2x2 + 1 (alt)
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(2,3),(0,4),(1,4)],
        # Set 44: 3x4 + 2
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(3,2)],
        # Set 45: 4x3 + 2 (alt)
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(0,2),(3,2)],
        # Set 46: 5x2 + 1 (center)
        [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(0,2)],
        # Set 47: 3x3 + 2 (side)
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(3,1),(-1,0)],
        # Set 48: 4x2 + 3 (alt)
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)],
        # Set 49: 2x5 + 1 (wide)
        [(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(4,1)],
        # Set 50: 3x4 + 1 (alt)
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(2,3)],
    ]

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def rotate_axial(axial_list, steps):
        """Rotate axial coords by steps * 60 degrees."""
        result = []
        for q, r in axial_list:
            for _ in range(steps % 6):
                q, r = -r, q + r
            result.append((q, r))
        return result

    # Generate all candidate point sets (base + 60-degree rotations)
    all_candidates = []
    seen = set()
    for base in base_sets:
        for rot in range(6):
            rotated = rotate_axial(base, rot)
            key = tuple(sorted(rotated))
            if key not in seen:
                seen.add(key)
                all_candidates.append(rotated)

    # ---- Fast exact objective: S = (max_proj + APOTHEM) / APOTHEM ----
    def fast_exact_S(pts, cx, cy, beta):
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        centered = pts - np.array([cx, cy])
        projs = centered @ normals.T
        max_proj = np.max(np.abs(projs))
        return (max_proj + APOTHEM) / APOTHEM

    # ---- Smooth max via log-sum-exp for gradient-friendly optimization ----
    def smooth_max(arr, temperature=0.01):
        """Approximate max using log-sum-exp."""
        arr = np.asarray(arr, dtype=np.float64)
        max_val = np.max(arr)
        if max_val > 1e10:
            return max_val
        shifted = arr - max_val
        log_sum = np.log(np.sum(np.exp(shifted / temperature))) * temperature
        return max_val + log_sum

    def smooth_exact_S(pts, cx, cy, beta, temp=0.01):
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        centered = pts - np.array([cx, cy])
        projs = centered @ normals.T
        abs_projs = np.abs(projs)
        # Smooth max over all 6*11 values
        flat = abs_projs.flatten()
        max_proj = smooth_max(flat, temp)
        return (max_proj + APOTHEM) / APOTHEM

    # ---- LP-based exact S for given beta (find optimal center) ----
    # The LP: minimize D
    # s.t. for each k in 0..5, for each point p:
    #   n_k . c - D <= n_k . p
    #   -n_k . c - D <= -n_k . p
    # Variables: [cx, cy, D]
    def calc_S_for_beta(pts, beta):
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        n_pts = len(pts)
        n_constraints = 6 * n_pts * 2

        A_ub = np.zeros((n_constraints, 3))
        b_ub = np.zeros(n_constraints)
        row = 0
        for k in range(6):
            nx, ny = normals[k]
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            for i in range(n_pts):
                # n_k . c - D <= n_k . p_i
                A_ub[row, 0] = nx
                A_ub[row, 1] = ny
                A_ub[row, 2] = -1.0
                b_ub[row] = projs[i]
                row += 1
                # -n_k . c - D <= -n_k . p_i
                A_ub[row, 0] = -nx
                A_ub[row, 1] = -ny
                A_ub[row, 2] = -1.0
                b_ub[row] = -projs[i]
                row += 1

        c_obj = np.array([0.0, 0.0, 1.0])
        bounds = [(None, None), (None, None), (0, None)]

        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                cx, cy = res.x[0], res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy)
            else:
                return float('inf'), None
        except Exception:
            return float('inf'), None

    # ---- Optimize a single point set ----
    def optimize_shape(pts, current_best_S=float('inf')):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None

        if time.time() - START_TIME > TIME_BUDGET - SAFETY_MARGIN:
            return float('inf'), 0.0, None

        # Phase 1: Coarse grid over beta (37 points)
        for i in range(37):
            if time.time() - START_TIME > TIME_BUDGET - SAFETY_MARGIN:
                break
            beta = i * (math.pi / 6) / 36
            S, center = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center

        # Early exit
        if best_S >= current_best_S * 1.001 and current_best_S < float('inf'):
            return best_S, best_beta, best_center

        # Phase 2: Bounded scalar refinement of beta
        try:
            if time.time() - START_TIME < TIME_BUDGET - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0],
                                     bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 200, 'xatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass

        # Phase 3: Multi-start Nelder-Mead on smooth objective
        try:
            if time.time() - START_TIME < TIME_BUDGET - SAFETY_MARGIN:
                def smooth_objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    return smooth_exact_S(pts, cx, cy, beta, temp=0.005)

                bcx, bcy = best_center
                bb = best_beta
                starts = [
                    [bcx, bcy, bb],
                    [bcx, bcy, bb + 0.02],
                    [bcx, bcy, bb - 0.02],
                    [bcx, bcy, bb + 0.05],
                    [bcx, bcy, bb - 0.05],
                    [bcx + 0.02, bcy, bb],
                    [bcx - 0.02, bcy, bb],
                    [bcx, bcy + 0.02, bb],
                    [bcx, bcy - 0.02, bb],
                    [bcx + 0.01, bcy + 0.01, bb],
                    [bcx - 0.01, bcy - 0.01, bb],
                    [bcx + 0.01, bcy - 0.01, bb],
                    [bcx - 0.01, bcy + 0.01, bb],
                    [bcx, bcy, 0.0],
                    [bcx, bcy, math.pi / 12],
                    [bcx, bcy, math.pi / 6],
                ]

                for init in starts:
                    if time.time() - START_TIME > TIME_BUDGET - SAFETY_MARGIN:
                        break
                    try:
                        res = minimize(smooth_objective, init, method='Nelder-Mead',
                                     options={'maxiter': 8000, 'xatol': 1e-13, 'fatol': 1e-13})
                        if res.fun < best_S:
                            best_S = res.fun
                            best_center = (res.x[0], res.x[1])
                            best_beta = ((res.x[2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    except Exception:
                        pass
        except Exception:
            pass

        # Phase 4: Final high-precision polish with exact objective
        try:
            if time.time() - START_TIME < TIME_BUDGET - SAFETY_MARGIN:
                def exact_objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    return fast_exact_S(pts, cx, cy, beta)

                bcx, bcy = best_center
                bb = best_beta
                polish_starts = [
                    [bcx, bcy, bb],
                    [bcx + 0.005, bcy, bb],
                    [bcx - 0.005, bcy, bb],
                    [bcx, bcy + 0.005, bb],
                    [bcx, bcy - 0.005, bb],
                    [bcx, bcy, bb + 0.01],
                    [bcx, bcy, bb - 0.01],
                    [bcx + 0.003, bcy + 0.003, bb],
                    [bcx - 0.003, bcy - 0.003, bb],
                ]
                for init in polish_starts:
                    if time.time() - START_TIME > TIME_BUDGET - SAFETY_MARGIN:
                        break
                    try:
                        res = minimize(exact_objective, init, method='Nelder-Mead',
                                     options={'maxiter': 15000, 'xatol': 1e-15, 'fatol': 1e-15})
                        if res.fun < best_S:
                            best_S = res.fun
                            best_center = (res.x[0], res.x[1])
                            best_beta = ((res.x[2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    except Exception:
                        pass
        except Exception:
            pass

        return best_S, best_beta, best_center

    # ---- Main loop over all candidates ----
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for idx, axial in enumerate(all_candidates):
        if time.time() - START_TIME > TIME_BUDGET - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts, current_best_S=best_global_S)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    # Fallback
    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(base_sets[0])
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

    # Tiny safety scale to avoid numerical overlap
    SCALE = 1.0 + 1e-12
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    pts_scaled = np.array(pts_scaled)

    # Recompute exact S with safety scale
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
