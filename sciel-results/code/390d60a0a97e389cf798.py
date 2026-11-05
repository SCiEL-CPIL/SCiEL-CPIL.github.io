import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s buffer for 150s limit
    
    n = 8
    num_pairs = n * (n - 1) // 2  # 28
    
    def compute_score(pts):
        """Calculate the actual score for the final output."""
        pts = np.array(pts, dtype=float)
        dists = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        iu = np.triu_indices(n, 1)
        d = dists[iu]
        D = np.max(d)
        if D < 1e-9:
            return -np.inf
        pts_scaled = pts / D * 2.0
        dists_scaled = np.linalg.norm(pts_scaled[:, np.newaxis, :] - pts_scaled[np.newaxis, :, :], axis=2)
        d_scaled = dists_scaled[iu]
        log_delta = np.sum(np.log(d_scaled))
        delta = np.exp(log_delta)
        score = delta / (n ** n)
        return score

    def two_squares(params):
        """
        Generate 8 points as two concentric squares.
        params = [r1, theta1, r2, theta2]
        Square 1: r1 * (cos(theta1 + k*pi/2), sin(theta1 + k*pi/2)) for k=0,1,2,3
        Square 2: r2 * (cos(theta2 + k*pi/2), sin(theta2 + k*pi/2)) for k=0,1,2,3
        """
        r1, theta1, r2, theta2 = params
        angles1 = theta1 + np.pi/2 * np.arange(4)
        angles2 = theta2 + np.pi/2 * np.arange(4)
        pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        return np.vstack([pts1, pts2])

    def neg_log_score_two_squares(params):
        """Negative log-score for the two-squares model (to be minimized)."""
        if time.time() > deadline:
            return 0.0
        pts = two_squares(params)
        dists = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        iu = np.triu_indices(n, 1)
        d = dists[iu]
        d = np.maximum(d, 1e-10)
        log_prod = np.sum(np.log(d))
        D = np.max(d)
        return 28 * np.log(D) - log_prod

    # Regular octagon: two squares with r1=r2=1, theta1=0, theta2=pi/4
    octagon = two_squares([1.0, 0.0, 1.0, np.pi/4])
    best_score = compute_score(octagon)
    best_pts = octagon

    # Grid search over the 4D parameter space
    # r1, r2 in [0.3, 1.5], theta1, theta2 in [0, pi/2] (symmetry: pi/2 periodicity)
    r_vals = np.linspace(0.3, 1.5, 30)
    theta_vals = np.linspace(0, np.pi/2, 20)
    
    best_grid_score = -np.inf
    best_grid_params = None
    
    for r1 in r_vals:
        for r2 in r_vals:
            for th1 in theta_vals:
                for th2 in theta_vals:
                    if time.time() > deadline - 30:
                        break
                    params = [r1, th1, r2, th2]
                    score = compute_score(two_squares(params))
                    if score > best_grid_score:
                        best_grid_score = score
                        best_grid_params = params
                if time.time() > deadline - 30:
                    break
            if time.time() > deadline - 30:
                break
        if time.time() > deadline - 30:
            break

    if best_grid_params is not None and best_grid_score > best_score:
        best_score = best_grid_score
        best_pts = two_squares(best_grid_params)

    # Local refinement from multiple starting points
    if time.time() < deadline - 15:
        # Starting points: best grid point, octagon, and some random points
        inits = []
        if best_grid_params is not None:
            inits.append(best_grid_params)
        inits.append([1.0, 0.0, 1.0, np.pi/4])  # Regular octagon
        
        rng = np.random.default_rng(42)
        for _ in range(20):
            r1 = rng.uniform(0.3, 1.5)
            r2 = rng.uniform(0.3, 1.5)
            th1 = rng.uniform(0, np.pi/2)
            th2 = rng.uniform(0, np.pi/2)
            inits.append([r1, th1, r2, th2])
        
        bounds = [(0.1, 2.0), (0, np.pi/2), (0.1, 2.0), (0, np.pi/2)]
        
        for init in inits:
            if time.time() > deadline - 10:
                break
            try:
                res = minimize(
                    neg_log_score_two_squares,
                    init,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12}
                )
                if res.success:
                    candidate_pts = two_squares(res.x)
                    cand_score = compute_score(candidate_pts)
                    if cand_score > best_score:
                        best_score = cand_score
                        best_pts = candidate_pts
            except Exception:
                pass

    # If time remains, also try a full 16D optimization from the best two-squares result
    if time.time() < deadline - 20:
        def neg_log_score_full(pts_flat):
            """Negative log-score for full 16D parameterization."""
            if time.time() > deadline:
                return 0.0
            pts = pts_flat.reshape(n, 2)
            dists = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
            iu = np.triu_indices(n, 1)
            d = dists[iu]
            d = np.maximum(d, 1e-10)
            log_prod = np.sum(np.log(d))
            D = np.max(d)
            return 28 * np.log(D) - log_prod

        # Start from the best two-squares result
        init_flat = best_pts.flatten()
        bounds_full = [(-2.0, 2.0)] * (2 * n)
        
        try:
            res = minimize(
                neg_log_score_full,
                init_flat,
                method='L-BFGS-B',
                bounds=bounds_full,
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            if res.success:
                candidate_pts = res.x.reshape(n, 2)
                cand_score = compute_score(candidate_pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = candidate_pts
        except Exception:
            pass

        # Also try from the regular octagon
        octagon_flat = two_squares([1.0, 0.0, 1.0, np.pi/4]).flatten()
        if time.time() < deadline - 15:
            try:
                res = minimize(
                    neg_log_score_full,
                    octagon_flat,
                    method='L-BFGS-B',
                    bounds=bounds_full,
                    options={'maxiter': 1000, 'ftol': 1e-12}
                )
                if res.success:
                    candidate_pts = res.x.reshape(n, 2)
                    cand_score = compute_score(candidate_pts)
                    if cand_score > best_score:
                        best_score = cand_score
                        best_pts = candidate_pts
            except Exception:
                pass

    return best_pts.tolist()
