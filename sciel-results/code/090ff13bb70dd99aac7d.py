import numpy as np
from scipy.optimize import minimize, differential_evolution, minimize_scalar
import time

def construct():
    start_time = time.time()
    timeout = 140.0  # 10s margin before 150s kill

    n = 8

    def log_product_exact(points):
        """
        Compute log product of pairwise distances after scaling to diameter 2.
        Uses exact max for diameter.
        """
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq)

        mask = ~np.eye(n, dtype=bool)
        off_diag_dists = dists[mask]

        if len(off_diag_dists) == 0:
            return -np.inf

        diameter = np.max(off_diag_dists)
        if diameter < 1e-10:
            return -np.inf

        scaled_dists = off_diag_dists * (2.0 / diameter)
        scaled_dists = np.maximum(scaled_dists, 1e-15)

        return np.sum(np.log(scaled_dists))

    def log_product_smooth(points, beta=100.0):
        """
        Compute log product of pairwise distances after scaling to diameter 2.
        Uses soft-max for diameter when beta is finite.
        """
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq)

        mask = ~np.eye(n, dtype=bool)
        off_diag_dists = dists[mask]

        if len(off_diag_dists) == 0:
            return -np.inf

        if beta < 1e6:
            scaled = off_diag_dists / beta
            m = np.max(scaled)
            log_sum_exp = m + np.log(np.sum(np.exp(scaled - m)))
            diameter = beta * log_sum_exp
        else:
            diameter = np.max(off_diag_dists)

        if diameter < 1e-10:
            return -np.inf

        scaled_dists = off_diag_dists * (2.0 / diameter)
        scaled_dists = np.maximum(scaled_dists, 1e-15)

        return np.sum(np.log(scaled_dists))

    def objective_smooth(x, beta=100.0):
        points = x.reshape(n, 2)
        lp = log_product_smooth(points, beta)
        return -lp

    def objective_exact(x):
        points = x.reshape(n, 2)
        lp = log_product_exact(points)
        return -lp

    # Baseline: Regular Octagon
    angles = 2 * np.pi * np.arange(n) / n
    reg_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    best_score = log_product_exact(reg_oct)
    best_points = reg_oct.copy()

    # Generate diverse initializations
    starts = [reg_oct.flatten()]

    # Two squares with various relative rotations
    for rot in np.linspace(0.0, 0.5, 20):
        a1 = 2 * np.pi * np.arange(4) / 4
        a2 = 2 * np.pi * (np.arange(4) / 4 + rot)
        sq1 = np.column_stack([np.cos(a1), np.sin(a1)])
        sq2 = np.column_stack([np.cos(a2), np.sin(a2)])
        starts.append(np.vstack([sq1, sq2]).flatten())

    # Antipodal pairs
    for seed_val in [0, 1, 2, 3, 4]:
        np.random.seed(seed_val)
        base_angles = np.sort(np.random.uniform(0, np.pi, 4))
        all_angles = np.concatenate([base_angles, base_angles + np.pi])
        pts = np.column_stack([np.cos(all_angles), np.sin(all_angles)])
        starts.append(pts.flatten())

    # Random points
    np.random.seed(42)
    for _ in range(20):
        pts = np.random.uniform(-1, 1, (n, 2))
        starts.append(pts.flatten())

    # Points on circle with perturbations
    for _ in range(10):
        base_angles = 2 * np.pi * np.arange(n) / n + np.random.uniform(-0.15, 0.15, n)
        radii = 1.0 + np.random.uniform(-0.15, 0.15, n)
        pts = np.column_stack([np.cos(base_angles) * radii, np.sin(base_angles) * radii])
        starts.append(pts.flatten())

    # Phase 1: Local optimization with soft-max surrogate (L-BFGS-B)
    for s in starts:
        if time.time() - start_time > timeout - 40:
            break
        try:
            res = minimize(
                lambda x: objective_smooth(x, beta=50.0),
                s,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < np.inf:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = res.x.reshape(n, 2)
        except Exception:
            pass

    # Phase 2: Refine with higher beta
    if time.time() - start_time < timeout - 30:
        for beta_val in [200.0, 500.0, 1000.0]:
            try:
                res = minimize(
                    lambda x: objective_smooth(x, beta=beta_val),
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if res.fun < np.inf:
                    score = -res.fun
                    if score > best_score:
                        best_score = score
                        best_points = res.x.reshape(n, 2)
            except Exception:
                pass

    # Phase 3: Global optimization with differential evolution
    if time.time() - start_time < timeout - 40:
        bounds = [(-5, 5)] * (2 * n)
        try:
            de_res = differential_evolution(
                lambda x: objective_smooth(x, beta=100.0),
                bounds,
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-10,
                polish=True
            )
            if de_res.fun < np.inf:
                score = -de_res.fun
                if score > best_score:
                    best_score = score
                    best_points = de_res.x.reshape(n, 2)
        except Exception:
            pass

    # Phase 4: High-precision polish with exact diameter
    if time.time() - start_time < timeout - 10:
        for method in ['L-BFGS-B', 'BFGS']:
            try:
                final_res = minimize(
                    objective_exact,
                    best_points.flatten(),
                    method=method,
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if final_res.fun < np.inf:
                    score = -final_res.fun
                    if score > best_score:
                        best_score = score
                        best_points = final_res.x.reshape(n, 2)
            except Exception:
                pass

    # Phase 5: Dedicated optimization for two-squares configuration
    if time.time() - start_time < timeout - 5:
        def two_squares_score(theta):
            a1 = 2 * np.pi * np.arange(4) / 4
            a2 = 2 * np.pi * (np.arange(4) / 4 + theta)
            sq1 = np.column_stack([np.cos(a1), np.sin(a1)])
            sq2 = np.column_stack([np.cos(a2), np.sin(a2)])
            pts = np.vstack([sq1, sq2])
            return -log_product_exact(pts)

        # Minimize over theta in [0, pi/4] by symmetry
        try:
            theta_res = minimize_scalar(
                two_squares_score,
                bounds=(0.0, np.pi/4),
                method='bounded',
                options={'xatol': 1e-12}
            )
            if theta_res.success:
                theta = theta_res.x
                a1 = 2 * np.pi * np.arange(4) / 4
                a2 = 2 * np.pi * (np.arange(4) / 4 + theta)
                sq1 = np.column_stack([np.cos(a1), np.sin(a1)])
                sq2 = np.column_stack([np.cos(a2), np.sin(a2)])
                pts = np.vstack([sq1, sq2])
                score = log_product_exact(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
        except Exception:
            pass

    # Final validation
    dists = np.sqrt(np.sum((best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-10:
        best_points = reg_oct

    # Scale/Check bounds
    max_coord = np.max(np.abs(best_points))
    if max_coord > 1e6:
        best_points = best_points / max_coord * 1e5

    return [[float(p[0]), float(p[1])] for p in best_points]
