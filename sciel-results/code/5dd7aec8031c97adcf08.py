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
TIME_BUDGET_PER_CANDIDATE = 4.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # --- Candidate point sets (axial coordinates), 11 unit hexagons ---
    candidates = []
    axial_1 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)]
    candidates.append(axial_1)
    axial_2 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)]
    candidates.append(axial_2)
    axial_3 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)]
    candidates.append(axial_3)
    axial_4 = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)]
    candidates.append(axial_4)
    axial_5 = [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)]
    candidates.append(axial_5)
    axial_6 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)]
    candidates.append(axial_6)
    axial_7 = [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)]
    candidates.append(axial_7)
    axial_8 = [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)]
    candidates.append(axial_8)
    axial_9 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)]
    candidates.append(axial_9)
    axial_10 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)]
    candidates.append(axial_10)
    axial_11 = [(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)]
    candidates.append(axial_11)
    axial_12 = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4)]
    candidates.append(axial_12)
    axial_13 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)]
    candidates.append(axial_13)
    axial_14 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)]
    candidates.append(axial_14)
    axial_15 = [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1),(0,-1)]
    candidates.append(axial_15)
    axial_16 = [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)]
    candidates.append(axial_16)
    axial_17 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)]
    candidates.append(axial_17)
    axial_18 = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2)]
    candidates.append(axial_18)
    axial_19 = [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)]
    candidates.append(axial_19)
    axial_20 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,1)]
    candidates.append(axial_20)
    axial_21 = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)]
    candidates.append(axial_21)
    axial_22 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)]
    candidates.append(axial_22)
    axial_23 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)]
    candidates.append(axial_23)
    axial_24 = [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4)]
    candidates.append(axial_24)
    axial_25 = [(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)]
    candidates.append(axial_25)
    axial_26 = [(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)]
    candidates.append(axial_26)
    axial_27 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,-2)]
    candidates.append(axial_27)
    axial_28 = [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(2,3),(1,4),(2,4)]
    candidates.append(axial_28)
    axial_29 = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)]
    candidates.append(axial_29)
    axial_30 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(-1,2)]
    candidates.append(axial_30)
    axial_31 = [(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3),(2,3)]
    candidates.append(axial_31)
    axial_32 = [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)]
    candidates.append(axial_32)
    axial_33 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(0,3),(1,-2)]
    candidates.append(axial_33)
    axial_34 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2),(0,3),(1,3)]
    candidates.append(axial_34)
    axial_35 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,2)]
    candidates.append(axial_35)
    axial_36 = [(0,0),(1,0),(2,0),(1,1),(2,1),(3,1),(0,2),(1,2),(0,3),(1,3),(2,3)]
    candidates.append(axial_36)
    axial_37 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(3,3),(0,4)]
    candidates.append(axial_37)
    axial_38 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,2),(-1,-2)]
    candidates.append(axial_38)
    axial_39 = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)]
    candidates.append(axial_39)
    axial_40 = [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(1,4),(2,4)]
    candidates.append(axial_40)
    axial_41 = [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(0,4)]
    candidates.append(axial_41)
    axial_42 = [(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4)]
    candidates.append(axial_42)
    axial_43 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)]
    candidates.append(axial_43)
    axial_44 = [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4),(1,4)]
    candidates.append(axial_44)
    axial_45 = [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)]
    candidates.append(axial_45)
    axial_46 = [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(0,4)]
    candidates.append(axial_46)
    axial_47 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3)]
    candidates.append(axial_47)
    axial_48 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)]
    candidates.append(axial_48)
    axial_49 = [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(1,4)]
    candidates.append(axial_49)
    axial_50 = [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)]
    candidates.append(axial_50)

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts, dtype=float)

    def convex_hull(points):
        """Andrew monotone chain. Returns hull vertices (no duplicates)."""
        pts = np.unique(np.round(points, 12), axis=0)
        if len(pts) <= 2:
            return pts
        order = np.lexsort((pts[:, 1], pts[:, 0]))
        p = pts[order]
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        lower = []
        for pt in p:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], pt) <= 0:
                lower.pop()
            lower.append(pt)
        upper = []
        for pt in p[::-1]:
            while len(upper) >= 2 and cross(upper[-2], upper[-1], pt) <= 0:
                upper.pop()
            upper.append(pt)
        hull = np.array(lower[:-1] + upper[:-1], dtype=float)
        return hull

    def calc_S_for_beta(pts, beta):
        """LP: minimize D s.t. |dot(p-c,n_k)|<=D for all hull pts p and 6 normals n_k.
        Returns (S, (cx,cy), D)."""
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        n_pts = len(pts)
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        for nx, ny in normals:
            for i in range(n_pts):
                px, py = pts[i]
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

    def exact_S(pts, beta, cx, cy):
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        pts_centered = pts - np.array([cx, cy])
        projs = np.abs(pts_centered @ normals.T)
        max_proj = float(np.max(projs))
        S = (max_proj + APOTHEM) / APOTHEM
        return S

    def smooth_calc_S(pts, beta, cx, cy, temperature=0.1):
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        max_val = np.max(all_projs)
        if max_val > 100:
            return float('inf')
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        D = temperature * log_sum
        S = (D + APOTHEM) / APOTHEM
        return S

    def newton_center_polish(pts, beta, cx, cy, D, tol=1e-13, max_iter=40):
        """2-D Newton on center using active (tied) support directions.
        Objective f(c)=max_i,k |dot(p_i-c,n_k)| is piecewise linear; at an optimum
        the active set spans R^2, so solving the linear system exactly refines c."""
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        for _ in range(max_iter):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            pc = pts - np.array([cx, cy])
            projs = pc @ normals.T  # (n,6)
            Dcur = float(np.max(np.abs(projs)))
            active = np.abs(projs) >= Dcur - tol
            rows = []
            rhs = []
            for i in range(len(pts)):
                for k in range(6):
                    if active[i, k]:
                        sgn = 1.0 if projs[i, k] >= 0 else -1.0
                        rows.append([sgn * normals[k, 0], sgn * normals[k, 1]])
                        rhs.append(sgn * (projs[i, k] - Dcur))
            if not rows:
                break
            A = np.array(rows)
            b = np.array(rhs)
            try:
                delta, *_ = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                break
            step = float(np.linalg.norm(delta))
            cx += delta[0]
            cy += delta[1]
            if step < 1e-14:
                break
        return cx, cy

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None

        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None

        # Coarse grid search over beta (exploit 30-degree symmetry: beta in [0, pi/6])
        n_grid = 60
        for i in range(n_grid + 1):
            beta = i * (math.pi / 6.0) / n_grid
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center

        # Bounded scalar refinement from several starting points
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                refine_starts = [best_beta, 0.0, math.pi/12, math.pi/6*0.25, math.pi/6*0.5, math.pi/6*0.75]
                for rb in refine_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    lo = max(0.0, rb - 0.25)
                    hi = min(math.pi/6.0, rb + 0.25)
                    res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0],
                                          bounds=(lo, hi), method='bounded',
                                          options={'maxiter': 400, 'xatol': 1e-14})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_beta = res.x
                        _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass

        # Smooth surrogate multi-start optimization
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                temperatures = [0.15, 0.08, 0.04, 0.02, 0.01]
                bcx, bcy = best_center
                start_configs = [
                    (bcx, bcy, best_beta),
                    (bcx, bcy, best_beta + 0.05),
                    (bcx, bcy, best_beta - 0.05),
                    (bcx, bcy, 0.0),
                    (bcx, bcy, math.pi/12),
                ]
                for temp in temperatures:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    for sc in start_configs:
                        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                            break
                        def smooth_obj(params, t=temp):
                            cx, cy, beta = params
                            beta = ((beta % (math.pi/6.0)) + math.pi/6.0) % (math.pi/6.0)
                            return smooth_calc_S(pts, beta, cx, cy, temperature=t)
                        res = minimize(smooth_obj, list(sc), method='Nelder-Mead',
                                       options={'maxiter': 800, 'xatol': 1e-13, 'fatol': 1e-13,
                                                'adaptive': True})
                        if res.fun < 1e10:
                            cx_r, cy_r, beta_r = res.x
                            beta_r = ((beta_r % (math.pi/6.0)) + math.pi/6.0) % (math.pi/6.0)
                            S_exact = exact_S(pts, beta_r, cx_r, cy_r)
                            if S_exact < best_S:
                                best_S = S_exact
                                best_beta = beta_r
                                best_center = (float(cx_r), float(cy_r))
        except Exception:
            pass

        # Exact Nelder-Mead polish with many perturbed starts
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6.0)) + math.pi/6.0) % (math.pi/6.0)
                    return exact_S(pts, beta, cx, cy)
                bcx, bcy = best_center
                bb = best_beta
                polish_starts = [[bcx, bcy, bb]]
                for dr in [0.03, -0.03, 0.01, -0.01]:
                    for ang in range(0, 360, 30):
                        dx = dr * math.cos(math.radians(ang))
                        dy = dr * math.sin(math.radians(ang))
                        polish_starts.append([bcx + dx, bcy + dy, bb])
                for db in [-0.02, -0.008, 0.008, 0.02, -0.003, 0.003]:
                    polish_starts.append([bcx, bcy, bb + db])
                for db in [-0.01, 0.01]:
                    for dr in [0.02, -0.02]:
                        polish_starts.append([bcx + dr, bcy, bb + db])
                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res = minimize(objective, init, method='Nelder-Mead',
                                   options={'maxiter': 12000, 'xatol': 1e-16, 'fatol': 1e-16,
                                            'adaptive': True})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (float(res.x[0]), float(res.x[1]))
                        best_beta = ((float(res.x[2]) % (math.pi/6.0)) + math.pi/6.0) % (math.pi/6.0)
        except Exception:
            pass

        # Newton center polish at the best orientation (rotating-calipers active set)
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                normals = np.array([
                    (math.cos(best_beta + math.pi / 6.0 + k * math.pi / 3.0),
                     math.sin(best_beta + math.pi / 6.0 + k * math.pi / 3.0))
                    for k in range(6)
                ])
                bcx, bcy = best_center
                pc = pts - np.array([bcx, bcy])
                Dcur = float(np.max(np.abs(pc @ normals.T)))
                ncx, ncy = newton_center_polish(pts, best_beta, bcx, bcy, Dcur)
                S_new = exact_S(pts, best_beta, ncx, ncy)
                if S_new < best_S:
                    best_S = S_new
                    best_center = (float(ncx), float(ncy))
        except Exception:
            pass

        # Ultra-fine final polish
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6.0)) + math.pi/6.0) % (math.pi/6.0)
                    return exact_S(pts, beta, cx, cy)
                bcx, bcy = best_center
                bb = best_beta
                ultra_starts = [
                    [bcx, bcy, bb],
                    [bcx + 0.005, bcy, bb], [bcx - 0.005, bcy, bb],
                    [bcx, bcy + 0.005, bb], [bcx, bcy - 0.005, bb],
                    [bcx, bcy, bb + 0.002], [bcx, bcy, bb - 0.002],
                ]
                for init in ultra_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res = minimize(objective, init, method='Nelder-Mead',
                                   options={'maxiter': 20000, 'xatol': 1e-16, 'fatol': 1e-16,
                                            'adaptive': True})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (float(res.x[0]), float(res.x[1]))
                        best_beta = ((float(res.x[2]) % (math.pi/6.0)) + math.pi/6.0) % (math.pi/6.0)
        except Exception:
            pass

        return best_S, best_beta, best_center

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    def hex_radius(axial_list):
        pts = axial_to_cartesian(axial_list)
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        max_proj = 0.0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
            max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
        return max_proj

    # Deduplicate candidate point sets and sort by compactness (process best first)
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = tuple(sorted(c))
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(c)

    sorted_indices = sorted(range(len(unique_candidates)),
                            key=lambda i: hex_radius(unique_candidates[i]))

    for idx in sorted_indices:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        axial = unique_candidates[idx]
        pts = axial_to_cartesian(axial)
        hull = convex_hull(pts)

        candidate_start = time.time()
        S, beta, center = optimize_shape(hull)
        elapsed = time.time() - candidate_start

        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

        if elapsed > TIME_BUDGET_PER_CANDIDATE:
            remaining = TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)
            if remaining < 20.0:
                break

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(unique_candidates[0])
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

    # Recompute the exact enclosing side length from the full point set (no scaling
    # that would risk overlap); the hexagons are disjoint by construction.
    normals = np.array([
        (math.cos(best_global_beta + math.pi / 6.0 + k * math.pi / 3.0),
         math.sin(best_global_beta + math.pi / 6.0 + k * math.pi / 3.0))
        for k in range(6)
    ])
    pc = best_global_pts - np.array([cx, cy])
    D_scaled = float(np.max(np.abs(pc @ normals.T)))
    S_scaled = (D_scaled + APOTHEM) / APOTHEM

    inner = []
    for x, y in best_global_pts:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)

    return inner, center, side, rotation
