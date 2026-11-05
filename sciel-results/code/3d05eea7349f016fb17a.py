import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def exact_score(P):
        """Compute the exact score for 8 points."""
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        i, j = np.triu_indices(8, k=1)
        dists = np.sqrt(dists_sq[i, j])
        if np.any(dists < 1e-12):
            return 0.0
        max_d = np.max(dists)
        if max_d < 1e-12:
            return 0.0
        log_prod = np.sum(np.log(dists))
        log_score = np.log(16.0) + log_prod - 28.0 * np.log(max_d)
        return np.exp(log_score)

    def neg_log_score_full(x):
        """Negative log score for 16-dim full parameterization (for minimization)."""
        P = x.reshape(8, 2)
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        i, j = np.triu_indices(8, k=1)
        dists = np.sqrt(dists_sq[i, j])
        if np.any(dists < 1e-12):
            return 1e10
        max_d = np.max(dists)
        if max_d < 1e-12:
            return 1e10
        log_prod = np.sum(np.log(dists))
        return -(np.log(16.0) + log_prod - 28.0 * np.log(max_d))

    def two_squares_to_points(params):
        """
        Convert 4 parameters to 8 points: two concentric squares.
        params = [r1, r2, theta1, theta2]
        Square 1: radius r1, rotation theta1
        Square 2: radius r2, rotation theta2
        """
        r1, r2, th1, th2 = params
        # Square 1 vertices at angles th1, th1+pi/2, th1+pi, th1+3pi/2
        a1 = np.array([th1, th1 + np.pi/2, th1 + np.pi, th1 + 3*np.pi/2])
        P1 = r1 * np.stack([np.cos(a1), np.sin(a1)], axis=1)
        # Square 2 vertices at angles th2, th2+pi/2, th2+pi, th2+3pi/2
        a2 = np.array([th2, th2 + np.pi/2, th2 + np.pi, th2 + 3*np.pi/2])
        P2 = r2 * np.stack([np.cos(a2), np.sin(a2)], axis=1)
        P = np.vstack([P1, P2])
        return P

    def neg_log_score_4d(params):
        """Negative log score for 4-param two-squares model."""
        P = two_squares_to_points(params)
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        i, j = np.triu_indices(8, k=1)
        dists = np.sqrt(dists_sq[i, j])
        if np.any(dists < 1e-12):
            return 1e10
        max_d = np.max(dists)
        if max_d < 1e-12:
            return 1e10
        log_prod = np.sum(np.log(dists))
        return -(np.log(16.0) + log_prod - 28.0 * np.log(max_d))

    def neg_log_score_polar_8d(params):
        """
        Negative log score for 8 points in polar coordinates.
        params = [r1, a1, r2, a2, ..., r8, a8]
        """
        r = params[0::2]
        a = params[1::2]
        P = np.stack([r * np.cos(a), r * np.sin(a)], axis=1)
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        i, j = np.triu_indices(8, k=1)
        dists = np.sqrt(dists_sq[i, j])
        if np.any(dists < 1e-12):
            return 1e10
        max_d = np.max(dists)
        if max_d < 1e-12:
            return 1e10
        log_prod = np.sum(np.log(dists))
        return -(np.log(16.0) + log_prod - 28.0 * np.log(max_d))

    # Phase 1: Grid search over the 4D two-squares manifold
    # r1, r2 in [0.2, 2.0], theta1, theta2 in [0, pi/2) (by symmetry)
    # The regular octagon corresponds to r1=r2=1, th1=0, th2=pi/4
    best_params_4d = np.array([1.0, 1.0, 0.0, np.pi/4])
    best_score_4d = exact_score(two_squares_to_points(best_params_4d))

    # Coarse grid
    r_vals = np.linspace(0.3, 2.0, 12)
    th1_vals = np.linspace(0.0, np.pi/2, 8)
    th2_vals = np.linspace(0.0, np.pi/2, 8)

    for r1 in r_vals:
        for r2 in r_vals:
            if time.time() > deadline - 60:
                break
            for th1 in th1_vals:
                for th2 in th2_vals:
                    params = np.array([r1, r2, th1, th2])
                    P = two_squares_to_points(params)
                    s = exact_score(P)
                    if s > best_score_4d:
                        best_score_4d = s
                        best_params_4d = params.copy()

    # Phase 2: Local refinement of the best 4D point using L-BFGS-B
    try:
        res4 = minimize(neg_log_score_4d, best_params_4d, method='L-BFGS-B',
                        bounds=[(0.1, 3.0), (0.1, 3.0), (0.0, np.pi/2), (0.0, np.pi/2)],
                        options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
        if res4.fun < neg_log_score_4d(best_params_4d):
            best_params_4d = res4.x
            P4 = two_squares_to_points(best_params_4d)
            s4 = exact_score(P4)
            if s4 > best_score_4d:
                best_score_4d = s4
    except Exception:
        pass

    # Phase 3: Convert to 16D and do a full-space polish with L-BFGS-B
    P_best = two_squares_to_points(best_params_4d)
    x16 = P_best.flatten()

    # Try multiple starting points for 16D polish
    candidates = [x16.copy()]
    # Also try regular octagon
    oct_pts = np.array([[np.cos(2*np.pi*k/8), np.sin(2*np.pi*k/8)] for k in range(8)])
    candidates.append(oct_pts.flatten())
    # Slightly perturbed versions
    rng = np.random.RandomState(42)
    for _ in range(3):
        pert = x16 + rng.normal(0, 0.05, size=16)
        candidates.append(pert)

    best_x16 = x16.copy()
    best_s16 = exact_score(P_best)

    for x0 in candidates:
        if time.time() > deadline - 20:
            break
        try:
            res16 = minimize(neg_log_score_full, x0, method='L-BFGS-B',
                             bounds=[(-5, 5)] * 16,
                             options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13})
            P16 = res16.x.reshape(8, 2)
            s16 = exact_score(P16)
            if s16 > best_s16:
                best_s16 = s16
                best_x16 = res16.x.copy()
        except Exception:
            pass

    # Phase 4: If time remains, try a short differential evolution from the best point
    if time.time() < deadline - 30:
        try:
            # Small local DE around the best solution
            bounds_de = []
            for i in range(16):
                lo = best_x16[i] - 0.5
                hi = best_x16[i] + 0.5
                bounds_de.append((lo, hi))
            res_de = differential_evolution(neg_log_score_full, bounds_de,
                                            maxiter=200, popsize=15, tol=1e-10,
                                            seed=42, polish=True, init='sobol')
            P_de = res_de.x.reshape(8, 2)
            s_de = exact_score(P_de)
            if s_de > best_s16:
                best_s16 = s_de
                best_x16 = res_de.x.copy()
        except Exception:
            pass

    # Final check
    P_final = best_x16.reshape(8, 2)
    final_score = exact_score(P_final)

    # Fallback to regular octagon if something went wrong
    if final_score < 0.99:
        P_final = oct_pts

    # Verify distinctness
    for i in range(8):
        for j in range(i+1, 8):
            if np.linalg.norm(P_final[i] - P_final[j]) < 1e-6:
                P_final = oct_pts
                break

    return [[float(P_final[i][0]), float(P_final[i][1])] for i in range(8)]
