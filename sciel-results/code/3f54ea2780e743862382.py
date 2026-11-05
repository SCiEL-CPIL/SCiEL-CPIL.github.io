import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n = 8

    def exact_score(points):
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq)
        mask = ~np.eye(n, dtype=bool)
        off_diag_dists = dists[mask]
        if len(off_diag_dists) == 0:
            return -np.inf
        diameter = np.max(off_diag_dists)
        if diameter < 1e-15:
            return -np.inf
        scaled_dists = off_diag_dists * (2.0 / diameter)
        scaled_dists = np.maximum(scaled_dists, 1e-15)
        log_prod = np.sum(np.log(scaled_dists))
        return log_prod - n * np.log(n)

    def log_product_smooth(points, beta):
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq)
        mask = ~np.eye(n, dtype=bool)
        off_diag_dists = dists[mask]
        if len(off_diag_dists) == 0:
            return np.inf
        scaled = off_diag_dists / beta
        m = np.max(scaled)
        log_sum_exp = m + np.log(np.sum(np.exp(scaled - m)))
        diameter = beta * log_sum_exp
        if diameter < 1e-15:
            return np.inf
        scaled_dists = off_diag_dists * (2.0 / diameter)
        scaled_dists = np.maximum(scaled_dists, 1e-15)
        log_prod = np.sum(np.log(scaled_dists))
        return n * np.log(n) - log_prod

    # Two-square parameterization
    def two_square_points(params):
        r1, a1, r2, a2 = params
        k = np.arange(4)
        a1_full = a1 + k * np.pi / 2
        a2_full = a2 + k * np.pi / 2
        sq1 = np.column_stack([np.cos(a1_full) * r1, np.sin(a1_full) * r1])
        sq2 = np.column_stack([np.cos(a2_full) * r2, np.sin(a2_full) * r2])
        return np.vstack([sq1, sq2])

    def two_square_obj(params, beta):
        pts = two_square_points(params)
        return log_product_smooth(pts, beta)

    # Baseline: regular octagon
    angles = 2 * np.pi * np.arange(n) / n
    reg_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    best_score_val = exact_score(reg_oct)
    best_points = reg_oct.copy()

    def try_update(points):
        nonlocal best_score_val, best_points
        if np.any(np.isnan(points)):
            return
        d = np.sqrt(np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :])**2, axis=2))
        np.fill_diagonal(d, np.inf)
        if np.min(d) < 1e-8:
            return
        s = exact_score(points)
        if s > best_score_val:
            best_score_val = s
            best_points = points.copy()

    # Phase 1: Two-square grid search + local optimization (fast, 4D)
    grid_starts = []
    grid_starts.append([1.0, 0.0, 1.0, np.pi/8])
    
    r1_vals = np.linspace(0.4, 1.4, 7)
    r2_vals = np.linspace(0.4, 1.4, 7)
    a2_vals = np.linspace(0.0, np.pi/2, 10)
    
    for r1 in r1_vals:
        for r2 in r2_vals:
            for a2 in a2_vals:
                grid_starts.append([r1, 0.0, r2, a2])
    
    for a1 in [0.0, np.pi/16, np.pi/8, np.pi/4]:
        for r1 in [0.6, 0.8, 1.0, 1.2]:
            for r2 in [0.6, 0.8, 1.0, 1.2]:
                for a2 in [0.0, np.pi/16, np.pi/8, np.pi/4, 3*np.pi/16]:
                    grid_starts.append([r1, a1, r2, a2])
    
    for gs in grid_starts:
        if time.time() > deadline - 50:
            break
        for beta in [50.0, 100.0]:
            if time.time() > deadline - 50:
                break
            try:
                res = minimize(
                    lambda x, b=beta: two_square_obj(x, b),
                    gs,
                    method='L-BFGS-B',
                    bounds=[(0.1, 5.0), (-np.pi, np.pi), (0.1, 5.0), (-np.pi, np.pi)],
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < np.inf:
                    candidate = two_square_points(res.x)
                    try_update(candidate)
            except Exception:
                pass

    # Phase 2: Differential evolution in full 16D space for global exploration
    if time.time() < deadline - 40:
        # Build diverse initial population
        pop_starts = [reg_oct.flatten()]
        
        # Two-square variants
        for rot in np.linspace(0.0, 0.49, 15):
            a1 = 2 * np.pi * np.arange(4) / 4
            a2 = 2 * np.pi * (np.arange(4) / 4 + rot)
            sq1 = np.column_stack([np.cos(a1), np.sin(a1)])
            sq2 = np.column_stack([np.cos(a2), np.sin(a2)])
            pop_starts.append(np.vstack([sq1, sq2]).flatten())
        
        # Antipodal pairs
        for seed_val in range(8):
            np.random.seed(seed_val)
            base_angles = np.sort(np.random.uniform(0, np.pi, 4))
            all_angles = np.concatenate([base_angles, base_angles + np.pi])
            pts = np.column_stack([np.cos(all_angles), np.sin(all_angles)])
            pop_starts.append(pts.flatten())
        
        # Random perturbations of octagon
        np.random.seed(999)
        for _ in range(20):
            jitter = np.random.normal(0, 0.3, (n, 2))
            pts = reg_oct + jitter
            pop_starts.append(pts.flatten())
        
        # Run DE with moderate budget
        bounds = [(-3, 3)] * (2 * n)
        try:
            de_res = differential_evolution(
                lambda x: log_product_smooth(x, 80.0),
                bounds,
                seed=42,
                maxiter=150,
                popsize=18,
                tol=1e-8,
                polish=True,
                updateinterval=10
            )
            if de_res.fun < np.inf:
                candidate = de_res.x.reshape(n, 2)
                try_update(candidate)
        except Exception:
            pass

    # Phase 3: Local refinement of best point in full 16D with increasing beta
    if time.time() < deadline - 20:
        for beta_val in [200.0, 500.0, 1000.0, 3000.0]:
            if time.time() > deadline - 10:
                break
            try:
                res = minimize(
                    lambda x: log_product_smooth(x.reshape(n, 2), beta_val),
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if res.fun < np.inf:
                    candidate = res.x.reshape(n, 2)
                    try_update(candidate)
            except Exception:
                pass

    # Phase 4: Project best point onto two-square manifold and refine there
    if time.time() < deadline - 15:
        proj_starts = [[1.0, 0.0, 1.0, np.pi/8]]
        
        # Try multiple splittings to fit two circles
        for split_seed in range(10):
            np.random.seed(split_seed)
            idx = np.random.permutation(n)
            g1 = idx[:4]
            g2 = idx[4:]
            try:
                from scipy.optimize import least_squares
                def fit_circle(points):
                    x, y = points[:, 0], points[:, 1]
                    def residuals(c):
                        cx, cy, r = c
                        return np.sqrt((x-cx)**2 + (y-cy)**2) - r
                    c0 = [np.mean(x), np.mean(y), 1.0]
                    res_c = least_squares(residuals, c0, max_nfev=100)
                    return res_c.x
                c1 = fit_circle(best_points[g1])
                c2 = fit_circle(best_points[g2])
                r1 = np.mean(np.sqrt((best_points[g1,0]-c1[0])**2 + (best_points[g1,1]-c1[1])**2))
                r2 = np.mean(np.sqrt((best_points[g2,0]-c2[0])**2 + (best_points[g2,1]-c2[1])**2))
                a1 = np.arctan2(c1[1], c1[0])
                a2 = np.arctan2(c2[1], c2[0])
                if r1 > 0.1 and r2 > 0.1:
                    proj_starts.append([r1, a1, r2, a2])
            except Exception:
                pass
        
        for ps in proj_starts:
            if time.time() > deadline - 8:
                break
            for beta_val in [500.0, 2000.0, 5000.0]:
                if time.time() > deadline - 5:
                    break
                try:
                    res = minimize(
                        lambda x, b=beta_val: two_square_obj(x, b),
                        ps,
                        method='L-BFGS-B',
                        bounds=[(0.1, 5.0), (-np.pi, np.pi), (0.1, 5.0), (-np.pi, np.pi)],
                        options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10}
                    )
                    if res.fun < np.inf:
                        candidate = two_square_points(res.x)
                        try_update(candidate)
                except Exception:
                    pass

    # Phase 5: Jitter + full 16D optimization to escape local minima
    if time.time() < deadline - 12:
        np.random.seed(42)
        for _ in range(12):
            if time.time() > deadline - 6:
                break
            for jitter_scale in [0.03, 0.08, 0.15]:
                if time.time() > deadline - 6:
                    break
                jitter = np.random.normal(0, jitter_scale, (n, 2))
                jittered = best_points + jitter
                try:
                    res = minimize(
                        lambda x: log_product_smooth(x.reshape(n, 2), 1000.0),
                        jittered.flatten(),
                        method='L-BFGS-B',
                        options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-9}
                    )
                    if res.fun < np.inf:
                        candidate = res.x.reshape(n, 2)
                        try_update(candidate)
                except Exception:
                    pass

    # Phase 6: Ultra-precise final polish
    if time.time() < deadline - 4:
        for beta_val in [10000.0, 50000.0, 200000.0]:
            if time.time() > deadline - 2:
                break
            try:
                final_res = minimize(
                    lambda x: log_product_smooth(x.reshape(n, 2), beta_val),
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 1500, 'ftol': 1e-16, 'gtol': 1e-14}
                )
                if final_res.fun < np.inf:
                    candidate = final_res.x.reshape(n, 2)
                    try_update(candidate)
            except Exception:
                pass

    # Final validation
    dists = np.sqrt(np.sum((best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-10:
        best_points = reg_oct

    max_coord = np.max(np.abs(best_points))
    if max_coord > 1e6:
        best_points = best_points / max_coord * 1e5

    return [[float(p[0]), float(p[1])] for p in best_points]
