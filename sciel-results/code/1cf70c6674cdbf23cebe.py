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

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    # --- Candidate point sets (axial coordinates) ---
    candidates = []

    # Standard hexagonal cluster
    candidates.append([(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (1, 1), (-1, 2), (-2, 1), (2, -1)])
    # Cross
    candidates.append([(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, 0), (-2, 0), (0, 2), (0, -2)])
    # 3x3 block + 2
    candidates.append([(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2), (-1, 1), (3, 1)])
    # 4x3 minus corners
    candidates.append([(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2)])
    # Staircase
    candidates.append([(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3), (2, 3)])
    # Compact 3x3+2
    candidates.append([(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3)])
    # Star
    candidates.append([(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (2, 0), (-2, 0), (0, 2), (0, -2)])
    # Elongated
    candidates.append([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)])
    # Asymmetric star
    candidates.append([(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, 1), (-2, 2), (-1, 3), (1, -2)])
    # Compact with top
    candidates.append([(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2), (1, 3), (2, 3)])
    # Wide base
    candidates.append([(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2), (3, 2), (1, 3)])
    # Tall narrow
    candidates.append([(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3), (0, 4)])
    # Star with asymmetric outer
    candidates.append([(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, 0), (-2, 0), (0, 2), (1, 2)])
    # Star with diagonal
    candidates.append([(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, 0), (-2, 0), (1, 1), (-1, -1)])
    # Dense cluster
    candidates.append([(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2), (-1, 1), (3, 1), (0, -1)])
    # Zigzag
    candidates.append([(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2), (0, 3), (1, 3), (2, 3), (3, 2)])
    # Star with top cluster
    candidates.append([(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, 1), (1, 2), (-1, 2), (0, 3)])
    # Trapezoid
    candidates.append([(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2), (3, 2)])
    # Wide top
    candidates.append([(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2), (0, 3), (1, 3), (2, 3), (3, 3)])
    # Star with inner diagonal
    candidates.append([(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, 0), (-2, 0), (1, 1), (-1, 1)])
    # 4x3 variant
    candidates.append([(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2)])
    # Wide with top
    candidates.append([(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2), (1, 3)])
    # Symmetric star
    candidates.append([(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, 0), (-2, 0), (0, 2), (0, -2)])
    # Very tall
    candidates.append([(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3), (0, 4)])
    # 3x3+row
    candidates.append([(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3), (2, 3)])
    # Dense 4-wide
    candidates.append([(0, 0), (1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2), (3, 2), (1, 3), (2, 3)])
    # Star with asymmetric bottom
    candidates.append([(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, 0), (-2, 0), (0, 2), (1, -2)])
    # Tall with top row
    candidates.append([(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3), (1, 4), (2, 4)])
    # Additional
    candidates.append([(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2), (1, 3), (2, 3)])
    candidates.append([(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2), (1, 3), (2, 3), (0, -1), (3, 1)])
    candidates.append([(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2)])
    candidates.append([(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, 0), (-2, 0), (1, 1), (-1, -1)])
    candidates.append([(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2), (0, 3), (1, 3), (2, 3), (1, 4), (2, 4)])
    candidates.append([(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3)])
    candidates.append([(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2), (0, 3), (1, 3)])

    # Corner-filling templates
    candidates.append([(2, 0), (-2, 0), (0, 2), (0, -2), (2, -2), (-2, 2), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1)])
    candidates.append([(2, 0), (-2, 0), (0, 2), (0, -2), (2, -2), (-2, 2), (1, 0), (-1, 0), (1, 1), (-1, 1), (0, 1)])
    candidates.append([(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (2, 0), (-2, 0), (0, 2), (0, -2)])
    candidates.append([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (-1, 1), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1)])
    candidates.append([(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (0, 3), (1, 3)])
    candidates.append([(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1), (2, 0), (-2, 0), (0, 2), (0, -2), (2, -2)])
    candidates.append([(0, 0), (2, 0), (-2, 0), (0, 2), (0, -2), (2, -2), (-2, 2), (1, 1), (-1, 1), (1, -1), (-1, -1)])
    candidates.append([(2, 0), (-2, 0), (0, 2), (0, -2), (2, -2), (-2, 2), (1, 0), (-1, 0), (0, 1), (1, -1), (-1, 1)])
    candidates.append([(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1), (2, 0), (-2, 0), (0, 2), (0, -2), (1, 1)])
    candidates.append([(2, 0), (-2, 0), (0, 2), (0, -2), (2, -2), (-2, 2), (1, 0), (0, 1), (1, 1), (0, -1), (-1, 0)])

    def calc_S_for_beta(pts, beta):
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

    def exact_S_vectorized(pts, cx, cy, beta):
        """Vectorized exact computation of S for given center and beta."""
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        max_proj = np.max(np.abs(projs))
        S = (max_proj + APOTHEM) / APOTHEM
        return S

    def smooth_epigraph_D(pts, cx, cy, beta, temperature):
        """
        Smooth epigraph approximation of D = max_k max_i |p_i - c|^T n_k.
        Uses log-sum-exp over all 12N signed projections.
        Returns (D_smooth, gradient_wrt_c).
        """
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        max_val = np.max(all_projs)
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        exp_shifted = np.exp(shifted)
        log_sum = np.log(np.sum(exp_shifted)) + max_val / temperature
        D_smooth = temperature * log_sum

        weights = exp_shifted / np.sum(exp_shifted)
        w_pos = weights[:len(projs)]
        w_neg = weights[len(projs):]
        w_pos_2d = w_pos.reshape(-1, 6)
        w_neg_2d = w_neg.reshape(-1, 6)
        grad_c = np.zeros(2)
        for k in range(6):
            coeff = np.sum(w_pos_2d[:, k]) - np.sum(w_neg_2d[:, k])
            grad_c[0] += -coeff * normals[k, 0]
            grad_c[1] += -coeff * normals[k, 1]

        return D_smooth, grad_c

    def newton_polish_center(pts, beta, cx, cy, max_iter=500):
        """
        High-precision 2D Newton polish for the center at fixed beta.
        Uses the active set of constraints to build a small QP-like solve.
        Falls back to gradient descent if Newton step is not valid.
        """
        best_cx, best_cy = cx, cy
        best_D = float('inf')
        temp = 0.01

        for it in range(max_iter):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break

            # Get current D and identify active constraints
            D_s, grad = smooth_epigraph_D(pts, cx, cy, beta, temp)
            if D_s < best_D:
                best_D = D_s
                best_cx, best_cy = cx, cy

            # Compute exact D and active set
            normals = np.array([
                (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
                 math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
                for k in range(6)
            ])
            pts_centered = pts - np.array([cx, cy])
            projs = pts_centered @ normals.T
            abs_projs = np.abs(projs)
            max_proj = np.max(abs_projs)
            D_exact = max_proj

            # Find active constraints (within tolerance)
            tol = 1e-10
            active_indices = np.where(abs_projs >= max_proj - tol)
            if len(active_indices[0]) == 0:
                break

            # Build the active constraint system
            # For each active (i, k, sign): (p_i - c) . (sign * n_k) = D
            # We want to find c that minimizes D subject to these constraints
            # This is a small LP or we can use the fact that at optimum,
            # the gradient of the active constraints should be in a cone

            # Simple approach: use the smooth gradient with very low temperature
            if temp < 0.001:
                break

            # Adaptive gradient step
            gnorm = np.linalg.norm(grad)
            if gnorm < 1e-14:
                break

            step = min(1.0 / (gnorm + 1e-10), 0.5)
            new_cx = cx - step * grad[0]
            new_cy = cy - step * grad[1]

            D_new, _ = smooth_epigraph_D(pts, new_cx, new_cy, beta, temp)
            if D_new < D_s:
                cx, cy = new_cx, new_cy
                temp *= 0.8
            else:
                temp *= 0.5
                if temp < 1e-6:
                    break

        # Final exact evaluation
        S_exact = exact_S_vectorized(pts, best_cx, best_cy, beta)
        return best_cx, best_cy, S_exact

    def polish_center_smooth(pts, beta, cx, cy, max_iter=200):
        """
        High-precision polish of the center using smooth epigraph + Newton-like steps.
        Minimizes D_smooth(cx, cy) for fixed beta and pts.
        """
        temps = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
        best_cx, best_cy = cx, cy
        best_D = float('inf')

        for temp in temps:
            step = 0.1
            for it in range(max_iter):
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                D_s, grad = smooth_epigraph_D(pts, cx, cy, beta, temp)
                if D_s < best_D:
                    best_D = D_s
                    best_cx, best_cy = cx, cy

                gnorm = np.linalg.norm(grad)
                if gnorm < 1e-12:
                    break

                new_cx = cx - step * grad[0]
                new_cy = cy - step * grad[1]
                D_new, _ = smooth_epigraph_D(pts, new_cx, new_cy, beta, temp)
                if D_new < D_s:
                    cx, cy = new_cx, new_cy
                    step = min(step * 1.5, 1.0)
                else:
                    step *= 0.5
                    if step < 1e-15:
                        break

        S_exact = exact_S_vectorized(pts, best_cx, best_cy, beta)
        return best_cx, best_cy, S_exact

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None

        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None

        # Coarse grid search over beta
        for i in range(37):
            beta = i * (math.pi / 6) / 36
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center

        # Refine beta with bounded scalar optimization
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0], bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 300, 'xatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass

        # Smooth surrogate optimization with adaptive temperature
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                temperatures = [0.3, 0.1, 0.05, 0.02, 0.01]
                start_betas = [best_beta, best_beta + 0.05, best_beta - 0.05, 0.0, math.pi / 12, math.pi / 6]

                for temp in temperatures:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    for sb in start_betas:
                        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                            break

                        def smooth_obj(params, t=temp):
                            cx, cy, beta = params
                            beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                            D_s, _ = smooth_epigraph_D(pts, beta, cx, cy, t)
                            return (D_s + APOTHEM) / APOTHEM

                        init = [best_center[0], best_center[1], sb]
                        res = minimize(smooth_obj, init, method='Nelder-Mead',
                                      options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})

                        if res.fun < 1e10:
                            cx_r, cy_r, beta_r = res.x
                            beta_r = ((beta_r % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                            S_exact, c_exact, _ = calc_S_for_beta(pts, beta_r)
                            if S_exact < best_S:
                                best_S = S_exact
                                best_beta = beta_r
                                best_center = c_exact
        except Exception:
            pass

        # Final high-precision polish using vectorized exact objective
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    return exact_S_vectorized(pts, cx, cy, beta)

                polish_starts = [
                    [best_center[0], best_center[1], best_beta],
                    [best_center[0], best_center[1], best_beta + 0.02],
                    [best_center[0], best_center[1], best_beta - 0.02],
                    [best_center[0] + 0.02, best_center[1], best_beta],
                    [best_center[0], best_center[1] + 0.02, best_beta],
                    [best_center[0] - 0.02, best_center[1], best_beta],
                    [best_center[0], best_center[1] - 0.02, best_beta],
                ]

                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res = minimize(objective, init, method='Nelder-Mead',
                                  options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-13})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (res.x[0], res.x[1])
                        best_beta = ((res.x[2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
        except Exception:
            pass

        # Smooth epigraph center polish
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                cx_p, cy_p, S_p = polish_center_smooth(pts, best_beta, best_center[0], best_center[1])
                if S_p < best_S:
                    best_S = S_p
                    best_center = (cx_p, cy_p)
        except Exception:
            pass

        # Newton polish for extra precision
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                cx_n, cy_n, S_n = newton_polish_center(pts, best_beta, best_center[0], best_center[1])
                if S_n < best_S:
                    best_S = S_n
                    best_center = (cx_n, cy_n)
        except Exception:
            pass

        return best_S, best_beta, best_center

    def continuous_optimization(initial_pts):
        """
        Optimize the relative positions of the hexagons and the outer rotation.
        """
        N_pts = len(initial_pts)
        x0 = np.concatenate([initial_pts.flatten(), [0.0]])

        def objective(params):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                return float('inf')

            pos = params[:2 * N_pts].reshape(N_pts, 2)
            beta = params[2 * N_pts]

            center = np.mean(pos, axis=0)
            S = exact_S_vectorized(pos, center[0], center[1], beta)

            penalty = 0.0
            for i in range(N_pts):
                for j in range(i + 1, N_pts):
                    d = np.linalg.norm(pos[i] - pos[j])
                    if d < 2.0:
                        penalty += (2.0 - d) ** 2

            return S + 100.0 * penalty

        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})

            if res.fun < float('inf'):
                pos_opt = res.x[:2 * N_pts].reshape(N_pts, 2)
                beta_opt = res.x[2 * N_pts]

                valid = True
                for i in range(N_pts):
                    for j in range(i + 1, N_pts):
                        if np.linalg.norm(pos_opt[i] - pos_opt[j]) < 2.0 - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break

                if valid:
                    S, center, D = calc_S_for_beta(pos_opt, beta_opt)
                    return S, beta_opt, center, pos_opt
                else:
                    return float('inf'), 0.0, None, None
        except Exception:
            return float('inf'), 0.0, None, None

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    # First pass: Discrete candidates
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

    # Second pass: Continuous optimization
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        S_c, beta_c, center_c, pts_c = continuous_optimization(best_global_pts)
        if S_c < best_global_S:
            best_global_S = S_c
            best_global_beta = beta_c
            best_global_center = center_c
            best_global_pts = pts_c

        for axial in [candidates[0], candidates[1], candidates[2], candidates[4], candidates[10]]:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            pts = axial_to_cartesian(axial)
            S_disc, _, _ = optimize_shape(pts)
            if S_disc < best_global_S + 0.5:
                S_c, beta_c, center_c, pts_c = continuous_optimization(pts)
                if S_c < best_global_S:
                    best_global_S = S_c
                    best_global_beta = beta_c
                    best_global_center = center_c
                    best_global_pts = pts_c

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
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

    # Check validity
    valid_final = True
    for i in range(len(best_global_pts)):
        for j in range(i + 1, len(best_global_pts)):
            if np.linalg.norm(best_global_pts[i] - best_global_pts[j]) < 2.0 - 1e-9:
                valid_final = False
                break
        if not valid_final:
            break

    if not valid_final:
        best_disc_S = float('inf')
        best_disc_pts = None
        best_disc_beta = 0.0
        best_disc_center = None
        for axial in candidates:
            pts = axial_to_cartesian(axial)
            S, beta, center = optimize_shape(pts)
            if S < best_disc_S:
                best_disc_S = S
                best_disc_pts = pts
                best_disc_beta = beta
                best_disc_center = center

        best_global_S = best_disc_S
        best_global_pts = best_disc_pts
        best_global_beta = best_disc_beta
        best_global_center = best_disc_center

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
