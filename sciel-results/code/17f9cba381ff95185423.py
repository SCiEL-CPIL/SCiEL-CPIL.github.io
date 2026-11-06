"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    n = 8
    iu = np.triu_indices(n, k=1)

    def log_score(pts):
        """Compute log of the score Delta/n^n where Delta is product of distances."""
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        d_ij = dists[iu]
        if np.any(d_ij < 1e-12):
            return -1e18
        log_delta = 2.0 * np.sum(np.log(d_ij))
        log_score_val = log_delta - n * np.log(n)
        return log_score_val

    def diameter(pts):
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        return np.max(dists)

    # Two concentric squares parameterization
    # Outer square: radius R, rotation phi
    # Inner square: radius r, rotation phi + pi/4 (to make 8 points)
    def two_squares_params_to_points(params):
        R, r, phi = params
        # Outer square at angles: phi, phi+pi/2, phi+pi, phi+3pi/2
        outer_angles = phi + np.pi/2 * np.arange(4)
        # Inner square at angles: phi + pi/4, phi + 3pi/4, phi + 5pi/4, phi + 7pi/4
        inner_angles = phi + np.pi/4 + np.pi/2 * np.arange(4)
        outer_pts = np.column_stack([R * np.cos(outer_angles), R * np.sin(outer_angles)])
        inner_pts = np.column_stack([r * np.cos(inner_angles), r * np.sin(inner_angles)])
        # Interleave to make 8 points
        pts = np.empty((8, 2))
        pts[0::2] = outer_pts
        pts[1::2] = inner_pts
        return pts

    def neg_score_two_squares(params):
        if time.time() > deadline - 5:
            return 1e18
        pts = two_squares_params_to_points(params)
        ls = log_score(pts)
        return -ls

    def neg_score_two_squares_constrained(params, penalty_weight):
        if time.time() > deadline - 5:
            return 1e18
        pts = two_squares_params_to_points(params)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        d_ij = dists[iu]
        if np.any(d_ij < 1e-12):
            return 1e18
        log_delta = 2.0 * np.sum(np.log(d_ij))
        log_score_val = log_delta - n * np.log(n)
        excess = np.maximum(d_ij - 2.0, 0.0)
        penalty = penalty_weight * np.sum(excess ** 2)
        return -log_score_val + penalty

    # General 8-point parameterization for full search
    def neg_score_general(x, penalty_weight):
        if time.time() > deadline - 5:
            return 1e18
        pts = x.reshape(n, 2)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        d_ij = dists[iu]
        if np.any(d_ij < 1e-12):
            return 1e18
        log_delta = 2.0 * np.sum(np.log(d_ij))
        log_score_val = log_delta - n * np.log(n)
        excess = np.maximum(d_ij - 2.0, 0.0)
        penalty = penalty_weight * np.sum(excess ** 2)
        return -log_score_val + penalty

    # Best tracking
    best_log_score = -np.inf
    best_points = None

    def try_update(pts):
        nonlocal best_log_score, best_points
        if pts is None:
            return
        ls = log_score(pts)
        if ls > best_log_score:
            best_log_score = ls
            best_points = pts.copy()

    # Initialize with regular octagon
    theta = 2 * np.pi * np.arange(n) / n
    points_oct = np.column_stack([np.cos(theta), np.sin(theta)])
    try_update(points_oct)

    # Phase 1: Grid search over two-square family
    # R in [0.8, 1.5], r in [0.0, 1.0], phi in [0, pi/4]
    R_vals = np.linspace(0.8, 1.5, 30)
    r_vals = np.linspace(0.0, 1.0, 25)
    phi_vals = np.linspace(0.0, np.pi/4, 20)

    for R in R_vals:
        if time.time() > deadline - 30:
            break
        for r in r_vals:
            if time.time() > deadline - 25:
                break
            for phi in phi_vals:
                if time.time() > deadline - 20:
                    break
                params = np.array([R, r, phi])
                pts = two_squares_params_to_points(params)
                try_update(pts)

    # Phase 2: Local refinement on best two-square params
    if best_points is not None and time.time() < deadline - 30:
        # Find the best params from the grid
        # Re-evaluate to find best params
        best_params = np.array([1.0, 0.0, 0.0])
        best_ps = -np.inf
        for R in R_vals:
            for r in r_vals:
                for phi in phi_vals:
                    params = np.array([R, r, phi])
                    pts = two_squares_params_to_points(params)
                    ps = log_score(pts)
                    if ps > best_ps:
                        best_ps = ps
                        best_params = params.copy()
        
        # Local refinement with Nelder-Mead
        try:
            res = minimize(neg_score_two_squares, best_params, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
            if -res.fun > best_log_score:
                best_log_score = -res.fun
                pts = two_squares_params_to_points(res.x)
                best_points = pts.copy()
        except Exception:
            pass

    # Phase 3: Full 8-point optimization from best two-square result
    if best_points is not None and time.time() < deadline - 15:
        x0 = best_points.flatten()
        for pw in [10.0, 50.0, 100.0]:
            if time.time() > deadline - 10:
                break
            try:
                result = minimize(
                    lambda x: neg_score_general(x, pw),
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                opt_pts = result.x.reshape(n, 2)
                try_update(opt_pts)
            except Exception:
                pass

    # Phase 4: Differential evolution for broader search
    if time.time() < deadline - 20:
        try:
            bounds = [(-2.0, 2.0)] * (2 * n)
            res = differential_evolution(
                neg_score_general,
                bounds,
                args=(100.0,),
                maxiter=100,
                popsize=20,
                tol=1e-10,
                seed=42,
                polish=True,
                updating='deferred',
                callback=lambda xk, convergence: None if time.time() < deadline - 5 else None
            )
            opt_pts = res.x.reshape(n, 2)
            try_update(opt_pts)
        except Exception:
            pass

    # Final: Scale to diameter exactly 2
    if best_points is not None:
        d = diameter(best_points)
        if d > 1e-10:
            best_points = best_points * (2.0 / d)
        
        # Ensure distinct points
        rng = np.random.RandomState(42)
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(best_points[i] - best_points[j])
                if dist < 1e-8:
                    best_points[j] += rng.randn(2) * 1e-4

    return [[float(p[0]), float(p[1])] for p in best_points]
# EVOLVE-BLOCK-END
