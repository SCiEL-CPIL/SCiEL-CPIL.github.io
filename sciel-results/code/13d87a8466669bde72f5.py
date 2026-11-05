import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill
    n = 8
    
    # Precompute mask for off-diagonal elements
    mask = ~np.eye(n, dtype=bool)
    
    def compute_score_log(points, beta):
        """
        Computes the log of the score: sum(log(dist)) - 56 * log(diameter/2).
        Uses soft-max for diameter if beta < 1e6.
        """
        # points: (8, 2)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq)
        
        off_diag_dists = dists[mask]
        
        if len(off_diag_dists) == 0:
            return -np.inf

        # Compute diameter
        if beta < 1e6:
            m = np.max(off_diag_dists)
            if m < 1e-10:
                return -np.inf
            diameter = m + (1.0 / beta) * np.log(np.sum(np.exp((off_diag_dists - m))))
        else:
            diameter = np.max(off_diag_dists)
            
        if diameter < 1e-10:
            return -np.inf

        log_dists_sum = np.sum(np.log(off_diag_dists))
        log_score_unnormalized = log_dists_sum - 56.0 * (np.log(diameter) - np.log(2.0))
        
        return log_score_unnormalized

    def objective(x, beta):
        points = x.reshape(n, 2)
        val = compute_score_log(points, beta)
        if np.isnan(val) or np.isinf(val):
            return 1e100
        return -val

    def two_square_points(r1, theta1, r2, theta2):
        """
        Generate 8 points as two concentric squares.
        Square 1: radius r1, rotation theta1
        Square 2: radius r2, rotation theta2
        """
        a1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2]) + theta1
        a2 = np.array([0, np.pi/2, np.pi, 3*np.pi/2]) + theta2
        sq1 = r1 * np.column_stack((np.cos(a1), np.sin(a1)))
        sq2 = r2 * np.column_stack((np.cos(a2), np.sin(a2)))
        return np.vstack([sq1, sq2])
    
    def two_square_score(params):
        """
        Compute log score for two-square configuration.
        params: [r1, theta1, r2, theta2]
        """
        r1, theta1, r2, theta2 = params
        if r1 < 1e-10 or r2 < 1e-10:
            return -1e100
        points = two_square_points(r1, theta1, r2, theta2)
        return compute_score_log(points, 1e8)

    # Regular octagon as baseline
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    p_oct = np.column_stack((np.cos(angles), np.sin(angles)))
    
    best_score = -np.inf
    best_points = p_oct
    
    # Phase 1: Grid search over two-square parameters
    # r1, r2 in [0.1, 2.0], theta1, theta2 in [0, pi/4] (by symmetry)
    r_vals = np.linspace(0.3, 1.8, 25)
    theta_vals = np.linspace(0, np.pi/4, 21)
    
    for r1 in r_vals:
        if time.time() > deadline - 60:
            break
        for r2 in r_vals:
            if time.time() > deadline - 60:
                break
            for theta1 in theta_vals:
                if time.time() > deadline - 60:
                    break
                for theta2 in theta_vals:
                    if time.time() > deadline - 60:
                        break
                    score = two_square_score([r1, theta1, r2, theta2])
                    if score > best_score:
                        best_score = score
                        best_points = two_square_points(r1, theta1, r2, theta2)
    
    # Phase 2: Local optimization in the 4D parameter space
    if time.time() < deadline - 40:
        # Extract best parameters from grid search
        # We need to find the best params from the grid
        best_params = None
        best_grid_score = -np.inf
        for r1 in r_vals:
            for r2 in r_vals:
                for theta1 in theta_vals:
                    for theta2 in theta_vals:
                        score = two_square_score([r1, theta1, r2, theta2])
                        if score > best_grid_score:
                            best_grid_score = score
                            best_params = [r1, theta1, r2, theta2]
        
        if best_params is not None:
            def param_objective(params):
                r1, theta1, r2, theta2 = params
                if r1 < 1e-8 or r2 < 1e-8:
                    return 1e100
                return -two_square_score(params)
            
            try:
                res = minimize(
                    param_objective,
                    best_params,
                    method='L-BFGS-B',
                    bounds=[(1e-4, None), (0, np.pi/2), (1e-4, None), (0, np.pi/2)],
                    options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if res.fun < 1e90:
                    score = -res.fun
                    if score > best_score:
                        best_score = score
                        best_points = two_square_points(*res.x)
            except Exception:
                pass
    
    # Phase 3: Full 16D optimization starting from best two-square config
    if time.time() < deadline - 20:
        beta_coarse = 50.0
        try:
            res = minimize(
                lambda x: objective(x, beta_coarse),
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < 1e90:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = res.x.reshape(n, 2)
        except Exception:
            pass
    
    # Phase 4: High precision refinement
    if time.time() < deadline - 10:
        beta_high = 200.0
        try:
            res = minimize(
                lambda x: objective(x, beta_high),
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.fun < 1e90:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = res.x.reshape(n, 2)
        except Exception:
            pass
    
    # Phase 5: Final polish with exact max
    if time.time() < deadline - 2:
        try:
            res = minimize(
                lambda x: objective(x, 1e8),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if res.fun < 1e90:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = res.x.reshape(n, 2)
        except Exception:
            pass

    # Validation
    dists = np.sqrt(np.sum((best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-10:
        best_points = p_oct

    # Scale to ensure coordinates are reasonable
    max_coord = np.max(np.abs(best_points))
    if max_coord > 1e6:
        best_points = best_points / max_coord * 1e5

    return [[float(p[0]), float(p[1])] for p in best_points]
