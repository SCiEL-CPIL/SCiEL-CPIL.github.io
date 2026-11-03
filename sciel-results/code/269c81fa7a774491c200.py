import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 12.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # --- Candidate point sets (axial coordinates) ---
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
        (0, 1), (1, 1), (2, 1),
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
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4), (2, 4)
    ]
    candidates.append(axial_28)

    axial_29 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_29)

    axial_30 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3)
    ]
    candidates.append(axial_30)

    axial_31 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_31)

    axial_32 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_32)

    axial_33 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_33)

    axial_34 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_34)

    axial_35 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_35)

    axial_36 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (1, 4)
    ]
    candidates.append(axial_36)

    axial_37 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3),
        (0, 3)
    ]
    candidates.append(axial_37)

    axial_38 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_38)

    axial_39 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3),
        (2, 3)
    ]
    candidates.append(axial_39)

    axial_40 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2),
        (0, 3),
        (3, 3)
    ]
    candidates.append(axial_40)

    axial_41 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3),
        (2, 3)
    ]
    candidates.append(axial_41)

    axial_42 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3),
        (2, 3)
    ]
    candidates.append(axial_42)

    axial_43 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3),
        (3, 3)
    ]
    candidates.append(axial_43)

    axial_44 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_44)

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def normals_for_beta(beta):
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        return np.column_stack([np.cos(angles), np.sin(angles)])

    def exact_D(pts, cx, cy, normals):
        centered = pts - np.array([cx, cy])
        projs = centered @ normals.T
        return float(np.max(np.abs(projs)))

    def calc_S_for_beta(pts, beta):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None

        normals = normals_for_beta(beta)

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
                cx = float(res.x[0])
                cy = float(res.x[1])
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def smooth_calc_S(pts, beta, temperature=0.1):
        # Smooth surrogate for the enclosing-hexagon half-width D = max|proj|,
        # then a small gradient-based polish on the center for high precision.
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None

        normals = normals_for_beta(beta)

        # initial center
        c0 = np.mean(pts, axis=0)

        def softmax_D(c):
            centered = pts - c
            projs = centered @ normals.T            # (n,6)
            allp = np.concatenate([projs.ravel(), (-projs).ravel()])
            m = np.max(allp)
            if m > 1e6:
                return float('inf')
            shifted = np.clip((allp - m) / temperature, -50.0, 50.0)
            lse = np.log(np.sum(np.exp(shifted)))
            return temperature * lse + m

        def softmax_D_and_grad(c):
            centered = pts - c
            projs = centered @ normals.T
            allp = np.concatenate([projs.ravel(), (-projs).ravel()])
            m = np.max(allp)
            if m > 1e6:
                return float('inf'), np.zeros(2)
            shifted = np.clip((allp - m) / temperature, -50.0, 50.0)
            w = np.exp(shifted)
            sw = np.sum(w)
            lse = np.log(sw)
            D = temperature * lse + m
            # gradient of D w.r.t. c:  dD/dc = -temperature * sum_j w_j/sum(w) * grad_c p_j
            # p_j = (pts[j]-c).dot(n_k)  ->  grad_c p_j = -n_k
            # so grad_c D = temperature * sum_j w_j/sum(w) * n_k(j)
            npts = len(pts)
            wpos = w[:npts * 6].reshape(npts, 6)
            wneg = w[npts * 6:].reshape(npts, 6)
            grad = temperature * (np.sum(wpos, axis=0) - np.sum(wneg, axis=0)) / sw
            grad = grad @ normals  # (6,) -> (2,)
            return D, grad

        best_c = c0
        best_val = softmax_D(c0)

        # refine center with L-BFGS-B on the smooth surrogate
        try:
            res = minimize(
                lambda c: softmax_D_and_grad(c)[0],
                c0,
                jac=lambda c: softmax_D_and_grad(c)[1],
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-12, 'ftol': 1e-12}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x
        except Exception:
            pass

        # report the EXACT value at the refined center
        D = exact_D(pts, best_c[0], best_c[1], normals)
        S = (D + APOTHEM) / APOTHEM
        return S, (float(best_c[0]), float(best_c[1])), D

    def optimize_shape(pts, current_best_S=float('inf')):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None

        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None

        # coarse rotation scan
        for i in range(37):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                return best_S, best_beta, best_center
            beta = i * (math.pi / 6) / 36
            S, center, D = smooth_calc_S(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center

        if best_S >= current_best_S * 1.001 and current_best_S < float('inf'):
            return best_S, best_beta, best_center

        # bounded scalar polish on rotation
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: smooth_calc_S(pts, b)[0],
                                      bounds=(0, math.pi / 6), method='bounded',
                                      options={'maxiter': 300, 'xatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = smooth_calc_S(pts, best_beta)
        except Exception:
            pass

        # final high-precision center polish at the chosen rotation
        try:
            if time.time() - TIME_LIMIT < TIME_LIMIT - SAFETY_MARGIN and best_center is not None:
                normals = normals_for_beta(best_beta)
                def obj(c):
                    return exact_D(pts, c[0], c[1], normals)
                # use smooth surrogate for gradient
                def obj_grad(c):
                    D, g = smooth_calc_S.__wrapped__ if False else (0, 0)
                    return D, g
                # simple: run L-BFGS-B on the smooth surrogate once more, tight
                def sobj(c):
                    return smooth_calc_S(pts, best_beta, temperature=0.02)[0]
                r2 = minimize(sobj, np.array(best_center), method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-12})
                if r2.fun < best_S * APOTHEM - APOTHEM:  # compare in D units
                    D2 = exact_D(pts, r2.x[0], r2.x[1], normals)
                    S2 = (D2 + APOTHEM) / APOTHEM
                    if S2 < best_S:
                        best_S = S2
                        best_center = (float(r2.x[0]), float(r2.x[1]))
        except Exception:
            pass

        return best_S, best_beta, best_center

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None
    best_candidates = []

    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts, current_best_S=best_global_S)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center
            best_candidates = [(S, pts, beta, center)]
        elif S < best_global_S * 1.01:
            best_candidates.append((S, pts, beta, center))

    # Local search on top candidates (position moves, fixed rotation)
    if best_global_pts is not None and time.time() < TIME_LIMIT - SAFETY_MARGIN:
        best_candidates.sort(key=lambda x: x[0])

        for S_cand, pts_cand, beta_cand, center_cand in best_candidates[:3]:
            if time.time() > TIME_LIMIT - SAFETY_MARGIN:
                break

            normals = normals_for_beta(beta_cand)
            min_dist = math.sqrt(3.0)

            def local_objective(params):
                pts_new = params.reshape(-1, 2)
                n = len(pts_new)
                # disjointness via Minkowski sum of two unit hexagons (same orientation)
                for i in range(n):
                    for j in range(i + 1, n):
                        diff = pts_new[i] - pts_new[j]
                        projs = diff @ normals.T
                        max_p = np.max(np.abs(projs))
                        if max_p < min_dist - 1e-7:
                            return 1e10 + (min_dist - max_p)
                c = np.mean(pts_new, axis=0)
                centered = pts_new - c
                projs_all = centered @ normals.T
                max_proj = np.max(np.abs(projs_all))
                return (max_proj + APOTHEM) / APOTHEM

            x0 = pts_cand.flatten()
            try:
                res = minimize(local_objective, x0, method='Nelder-Mead',
                               options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-10})
                if res.fun < best_global_S and res.fun < 1e9:
                    # recompute exact S at the optimizer's points
                    new_pts = res.x.reshape(-1, 2)
                    c = np.mean(new_pts, axis=0)
                    D = exact_D(new_pts, c[0], c[1], normals)
                    S2 = (D + APOTHEM) / APOTHEM
                    if S2 < best_global_S:
                        best_global_S = S2
                        best_global_pts = new_pts
                        best_global_center = (float(c[0]), float(c[1]))
                        best_global_beta = beta_cand
            except Exception:
                pass

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        normals = normals_for_beta(best_global_beta)
        D = exact_D(best_global_pts, best_global_center[0], best_global_center[1], normals)
        best_global_S = (D + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # exact enclosing size for the chosen configuration
    normals = normals_for_beta(best_global_beta)
    D_scaled = exact_D(best_global_pts, cx, cy, normals)
    S_scaled = (D_scaled + APOTHEM) / APOTHEM

    inner = []
    for x, y in best_global_pts:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)

    return inner, center, side, rotation
