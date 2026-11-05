# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin before 150s kill

    n = 8

    def exact_score(points):
        """Compute log product of pairwise distances after scaling to diameter 2."""
        pts = np.asarray(points, dtype=float)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq)
        mask = ~np.eye(n, dtype=bool)
        off_diag = dists[mask]
        if len(off_diag) == 0:
            return -np.inf
        diam = np.max(off_diag)
        if diam < 1e-12:
            return -np.inf
        scaled = off_diag * (2.0 / diam)
        scaled = np.maximum(scaled, 1e-15)
        return np.sum(np.log(scaled))

    def log_product_smooth(points, beta=100.0):
        """Smooth log product with soft-max diameter."""
        pts = np.asarray(points, dtype=float)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq)
        mask = ~np.eye(n, dtype=bool)
        off_diag = dists[mask]
        if len(off_diag) == 0:
            return -np.inf
        if beta < 1e6:
            scaled = off_diag / beta
            m = np.max(scaled)
            log_sum_exp = m + np.log(np.sum(np.exp(scaled - m)))
            diameter = beta * log_sum_exp
        else:
            diameter = np.max(off_diag)
        if diameter < 1e-10:
            return -np.inf
        scaled_dists = off_diag * (2.0 / diameter)
        scaled_dists = np.maximum(scaled_dists, 1e-15)
        return np.sum(np.log(scaled_dists))

    def objective_smooth(x, beta=100.0):
        points = x.reshape(n, 2)
        return -log_product_smooth(points, beta)

    def objective_exact(x):
        points = x.reshape(n, 2)
        return -exact_score(points)

    # Baseline: Regular Octagon
    angles = 2 * np.pi * np.arange(n) / n
    reg_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    best_score = exact_score(reg_oct)
    best_points = reg_oct.copy()

    # Generate diverse initializations
    starts = [reg_oct.flatten()]

    # Antipodal pairs with varying angles
    for seed_val in range(20):
        np.random.seed(seed_val)
        base_angles = np.sort(np.random.uniform(0, np.pi, 4))
        all_angles = np.concatenate([base_angles, base_angles + np.pi])
        pts = np.column_stack([np.cos(all_angles), np.sin(all_angles)])
        starts.append(pts.flatten())

    # Two squares with relative rotation
    for rot in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
        a1 = 2 * np.pi * np.arange(4) / 4
        a2 = 2 * np.pi * (np.arange(4) / 4 + rot)
        sq1 = np.column_stack([np.cos(a1), np.sin(a1)])
        sq2 = np.column_stack([np.cos(a2), np.sin(a2)])
        starts.append(np.vstack([sq1, sq2]).flatten())

    # Antipodal with varying radii
    for rad_pattern in [[1,1,1,1], [1,0.95,1,0.95], [1,0.9,1,0.9], [1,0.85,1,0.85],
                        [1.05,0.95,1.05,0.95], [1.1,0.9,1.1,0.9], [0.9,1.1,0.9,1.1]]:
        base_angles = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4])
        all_angles = np.concatenate([base_angles, base_angles + np.pi])
        radii = np.array(rad_pattern + rad_pattern)
        pts = np.column_stack([np.cos(all_angles) * radii, np.sin(all_angles) * radii])
        starts.append(pts.flatten())

    # Perturbed regular octagon
    np.random.seed(123)
    for _ in range(20):
        pert_angles = 2 * np.pi * np.arange(n) / n + np.random.uniform(-0.15, 0.15, n)
        pert_radii = 1.0 + np.random.uniform(-0.15, 0.15, n)
        pts = np.column_stack([np.cos(pert_angles) * pert_radii, np.sin(pert_angles) * pert_radii])
        starts.append(pts.flatten())

    # Random points
    np.random.seed(42)
    for _ in range(15):
        pts = np.random.uniform(-1, 1, (n, 2))
        starts.append(pts.flatten())

    # Phase 1: Local optimization with soft-max surrogate
    for s in starts:
        if time.time() - start_time > timeout - 40:
            break
        try:
            res = minimize(
                lambda x, b=80.0: objective_smooth(x, beta=b),
                s,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.fun < np.inf:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = res.x.reshape(n, 2)
        except Exception:
            pass

    # Phase 2: Refine with increasing beta
    if time.time() - start_time < timeout - 30:
        for beta_val in [200.0, 500.0, 1000.0, 5000.0]:
            if time.time() - start_time > timeout - 30:
                break
            try:
                res = minimize(
                    lambda x, b=beta_val: objective_smooth(x, beta=b),
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if res.fun < np.inf:
                    score = -res.fun
                    if score > best_score:
                        best_score = score
                        best_points = res.x.reshape(n, 2)
            except Exception:
                pass

    # Phase 3: Global optimization with differential evolution
    if time.time() - start_time < timeout - 35:
        bounds = [(-5, 5)] * (2 * n)
        try:
            de_res = differential_evolution(
                lambda x: objective_smooth(x, beta=100.0),
                bounds,
                seed=42,
                maxiter=500,
                popsize=20,
                tol=1e-12,
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
            if time.time() - start_time > timeout - 10:
                break
            try:
                final_res = minimize(
                    lambda x: objective_exact(x),
                    best_points.flatten(),
                    method=method,
                    options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14}
                )
                if final_res.fun < np.inf:
                    score = -final_res.fun
                    if score > best_score:
                        best_score = score
                        best_points = final_res.x.reshape(n, 2)
            except Exception:
                pass

    # Phase 5: Coordinate descent refinement
    if time.time() - start_time < timeout - 3:
        pts = best_points.copy()
        step_sizes = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        for step in step_sizes:
            if time.time() - start_time > timeout - 3:
                break
            improved = True
            while improved and time.time() - start_time < timeout - 3:
                improved = False
                for i in range(n):
                    old_pt = pts[i].copy()
                    for dx in [-step, step]:
                        for dy in [-step, step]:
                            new_pt = old_pt + np.array([dx, dy])
                            pts[i] = new_pt
                            s = exact_score(pts)
                            if s > best_score:
                                best_score = s
                                improved = True
                            else:
                                pts[i] = old_pt
        best_points = pts

    # Validation
    dists = np.sqrt(np.sum((best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-10:
        best_points = reg_oct

    # Scale/Check bounds
    max_coord = np.max(np.abs(best_points))
    if max_coord > 1e6:
        best_points = best_points / max_coord * 1e5

    return [[float(p[0]), float(p[1])] for p in best_points]
# EVOLVE-BLOCK-END
