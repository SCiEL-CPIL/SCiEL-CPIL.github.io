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
TIME_BUDGET_PER_CANDIDATE = 3.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    candidates = []

    axial_1 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)]
    axial_2 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)]
    axial_3 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)]
    axial_4 = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)]
    axial_5 = [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)]
    axial_6 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)]
    axial_7 = [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)]
    axial_8 = [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)]
    axial_9 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)]
    axial_10 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)]
    axial_11 = [(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)]
    axial_12 = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4)]
    axial_13 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)]
    axial_14 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)]
    axial_15 = [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1),(0,-1)]
    axial_16 = [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)]
    axial_17 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)]
    axial_18 = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2)]
    axial_19 = [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)]
    axial_20 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,1)]
    axial_21 = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)]
    axial_22 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)]
    axial_23 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)]
    axial_24 = [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4)]
    axial_25 = [(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)]
    axial_26 = [(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)]
    axial_27 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,-2)]
    axial_28 = [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(2,3),(1,4),(2,4)]
    axial_29 = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)]
    axial_30 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(-1,2)]
    axial_31 = [(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3),(2,3)]
    axial_32 = [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)]
    axial_33 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(0,3),(1,-2)]
    axial_34 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2),(0,3),(1,3)]
    axial_35 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,2)]
    axial_36 = [(0,0),(1,0),(2,0),(1,1),(2,1),(3,1),(0,2),(1,2),(0,3),(1,3),(2,3)]
    axial_37 = [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(3,3),(0,4)]
    axial_38 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,2),(-1,-2)]
    axial_39 = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)]
    axial_40 = [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(1,4),(2,4)]
    axial_41 = [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(0,4)]
    axial_42 = [(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4)]
    axial_43 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)]
    axial_44 = [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4),(1,4)]
    axial_45 = [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)]
    axial_46 = [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(0,4)]
    axial_47 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3)]
    axial_48 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)]
    axial_49 = [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(1,4)]
    axial_50 = [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)]
    for a in [axial_1,axial_2,axial_3,axial_4,axial_5,axial_6,axial_7,axial_8,axial_9,axial_10,
              axial_11,axial_12,axial_13,axial_14,axial_15,axial_16,axial_17,axial_18,axial_19,axial_20,
              axial_21,axial_22,axial_23,axial_24,axial_25,axial_26,axial_27,axial_28,axial_29,axial_30,
              axial_31,axial_32,axial_33,axial_34,axial_35,axial_36,axial_37,axial_38,axial_39,axial_40,
              axial_41,axial_42,axial_43,axial_44,axial_45,axial_46,axial_47,axial_48,axial_49,axial_50]:
        candidates.append(a)

    # A few additional compact topologies (new)
    extra = [
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)],  # dup ok
        [(0,0),(1,0),(2,0),(3,0),(4,0),(1,1),(2,1),(3,1),(4,1),(2,2),(3,2)],
        [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2)],
        [(0,0),(1,0),(0,1),(1,1),(-1,1),(2,1),(0,2),(1,2),(-1,2),(2,2),(0,3)],
        [(0,0),(1,0),(2,0),(3,0),(4,0),(5,0),(1,1),(2,1),(3,1),(4,1),(5,1)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(0,3)],
    ]
    for a in extra:
        if len(a) == N:
            candidates.append(a)

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def normals_for_beta(beta):
        return np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])

    def calc_S_for_beta(pts, beta):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
        normals = normals_for_beta(beta)
        n_pts = len(pts)
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        for nx, ny in normals:
            for i in range(n_pts):
                px, py = pts[i]
                A_ub.append([-nx, -ny, -1.0]); b_ub.append(-(px * nx + py * ny))
                A_ub.append([nx, ny, -1.0]);   b_ub.append(px * nx + py * ny)
        A_ub = np.array(A_ub); b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]; cx = res.x[0]; cy = res.x[1]
                return (D + APOTHEM) / APOTHEM, (cx, cy), D
            return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def exact_S(pts, beta, cx, cy):
        normals = normals_for_beta(beta)
        projs = np.abs((pts - np.array([cx, cy])) @ normals.T)
        return (float(np.max(projs)) + APOTHEM) / APOTHEM

    def smooth_calc_S(pts, beta, cx, cy, temperature=0.05):
        normals = normals_for_beta(beta)
        projs = (pts - np.array([cx, cy])) @ normals.T
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        max_val = float(np.max(all_projs))
        if max_val > 100:
            return float('inf')
        shifted = np.clip((all_projs - max_val) / temperature, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        return (temperature * log_sum + APOTHEM) / APOTHEM

    def active_set_refine(pts, cx0, cy0, beta0, max_iter=60):
        cx, cy = cx0, cy0
        beta = ((beta0 % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
        for _ in range(max_iter):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            normals = normals_for_beta(beta)
            pc = pts - np.array([cx, cy])
            projs = np.abs(pc @ normals.T)
            D = float(np.max(projs))
            tol = 1e-9
            active = np.argwhere(projs > D - tol)
            if len(active) < 2:
                break
            A = []; b = []
            for i, k in active:
                nx, ny = normals[k]; px, py = pts[i]
                proj_val = (px - cx) * nx + (py - cy) * ny
                sign = 1.0 if proj_val >= 0 else -1.0
                A.append([sign * nx, sign * ny, 1.0])
                b.append(sign * (nx * px + ny * py))
            A = np.array(A); b = np.array(b)
            try:
                sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                cx_new, cy_new, D_new = sol
                pc_new = pts - np.array([cx_new, cy_new])
                maxc = float(np.max(np.abs(pc_new @ normals.T)))
                if maxc <= D_new + 1e-9:
                    cx, cy, D = cx_new, cy_new, D_new
                    if abs(D - D_new) < 1e-14 and abs(cx - cx_new) < 1e-14 and abs(cy - cy_new) < 1e-14:
                        break
                else:
                    break
            except Exception:
                break
        return (D + APOTHEM) / APOTHEM, (cx, cy), beta

    def optimize_shape(pts):
        best_S = float('inf'); best_beta = 0.0; best_center = None
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None

        # Finer coarse grid over beta (period pi/6)
        n_grid = 60
        grid_results = []
        for i in range(n_grid + 1):
            beta = i * (math.pi / 6) / n_grid
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S; best_beta = beta; best_center = center
            grid_results.append((S, beta, center))
        grid_results.sort()

        # Multi-start bounded refinement from the best few grid points
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                starts = [best_beta] + [g[1] for g in grid_results[:4]]
                for rb in starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    lo = max(0.0, rb - 0.15); hi = min(math.pi / 6, rb + 0.15)
                    res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0],
                                          bounds=(lo, hi), method='bounded',
                                          options={'maxiter': 400, 'xatol': 1e-14})
                    if res.fun < best_S:
                        best_S = res.fun; best_beta = float(res.x)
                        _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass

        # Active-set refinement from best + small perturbations
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN and best_center is not None:
                bcx, bcy = best_center
                for dcx in (-0.02, 0.0, 0.02):
                    for dcy in (-0.02, 0.0, 0.02):
                        for db in (-0.02, 0.0, 0.02):
                            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                                break
                            try:
                                S_as, c_as, b_as = active_set_refine(pts, bcx + dcx, bcy + dcy, best_beta + db, max_iter=60)
                                if S_as < best_S:
                                    best_S = S_as; best_center = c_as; best_beta = b_as
                            except Exception:
                                pass
        except Exception:
            pass

        # Smooth polish (log-sum-exp surrogate of the nonsmooth max)
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN and best_center is not None:
                def smooth_obj(params, t=0.05):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    return smooth_calc_S(pts, beta, cx, cy, temperature=t)
                bcx, bcy = best_center
                starts = [
                    [bcx, bcy, best_beta],
                    [bcx + 0.03, bcy, best_beta], [bcx - 0.03, bcy, best_beta],
                    [bcx, bcy + 0.03, best_beta], [bcx, bcy - 0.03, best_beta],
                    [bcx, bcy, best_beta + 0.03], [bcx, bcy, best_beta - 0.03],
                ]
                for sc in starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    try:
                        res = minimize(smooth_obj, sc, method='Nelder-Mead',
                                       options={'maxiter': 300, 'xatol': 1e-11, 'fatol': 1e-11, 'adaptive': True})
                        if res.fun < 1e10:
                            cx_r, cy_r, beta_r = res.x
                            beta_r = ((beta_r % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                            S_exact = exact_S(pts, beta_r, cx_r, cy_r)
                            if S_exact < best_S:
                                best_S = S_exact; best_center = (float(cx_r), float(cy_r)); best_beta = beta_r
                    except Exception:
                        pass
        except Exception:
            pass

        # Final exact active-set polish from the smoothed result
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN and best_center is not None:
                bcx, bcy = best_center
                S_as, c_as, b_as = active_set_refine(pts, bcx, bcy, best_beta, max_iter=80)
                if S_as < best_S:
                    best_S = S_as; best_center = c_as; best_beta = b_as
        except Exception:
            pass

        return best_S, best_beta, best_center

    def hex_radius(axial_list):
        pts = axial_to_cartesian(axial_list)
        cx = float(np.mean(pts[:, 0])); cy = float(np.mean(pts[:, 1]))
        max_proj = 0.0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
            max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
        return max_proj

    # Deduplicate by frozenset of rounded coords
    seen = set(); uniq = []
    for c in candidates:
        key = frozenset((round(x, 3), round(y, 3)) for x, y in axial_to_cartesian(c))
        if key not in seen:
            seen.add(key); uniq.append(c)
    candidates = uniq

    sorted_indices = sorted(range(len(candidates)), key=lambda i: hex_radius(candidates[i]))

    best_global_S = float('inf'); best_global_pts = None
    best_global_beta = 0.0; best_global_center = None

    for idx in sorted_indices:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        remaining = TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)
        if remaining < 4.0:
            break
        pts = axial_to_cartesian(candidates[idx])
        t0 = time.time()
        S, beta, center = optimize_shape(pts)
        el = time.time() - t0
        if S < best_global_S:
            best_global_S = S; best_global_pts = pts; best_global_beta = beta; best_global_center = center
        if el > TIME_BUDGET_PER_CANDIDATE:
            if (TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)) < 12.0:
                break

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # Tiny inward scale to guarantee strict non-overlap of touching hexes
    SCALE = 1.0 - 1e-10
    pts_scaled = np.array([
        (cx + (x - cx) * SCALE, cy + (y - cy) * SCALE) for x, y in best_global_pts
    ])
    max_proj = 0.0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
    S_scaled = (max_proj + APOTHEM) / APOTHEM

    inner = [[float(x), float(y), float(angle_deg)] for x, y in pts_scaled]
    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)
    return inner, center, side, rotation
