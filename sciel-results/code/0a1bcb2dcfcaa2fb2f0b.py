import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s kill

    n = 8
    # Number of ordered pairs (i != j) is n*(n-1) = 56
    num_pairs = n * (n - 1)

    def exact_score(points):
        """
        Compute the exact score: product of distances over ordered pairs,
        after scaling so diameter = 2, then divide by n^n.
        Returns the log of the score (for optimization), or the actual score.
        """
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq)
        
        # Mask diagonal
        mask = ~np.eye(n, dtype=bool)
        off_diag_dists = dists[mask]
        
        if len(off_diag_dists) == 0:
            return -np.inf
        
        diameter = np.max(off_diag_dists)
        if diameter < 1e-15:
            return -np.inf
        
        # Scale to diameter 2
        scaled_dists = off_diag_dists * (2.0 / diameter)
        
        # Avoid log(0)
        scaled_dists = np.maximum(scaled_dists, 1e-15)
        
        log_prod = np.sum(np.log(scaled_dists))
        # The score is Delta / n^n, so log_score = log(Delta) - n*log(n)
        # The task says "score is Delta/n^n", and regular octagon scores 1.
        # Let's check: for regular octagon, diameter is 2 (if scaled to radius 1, diameter is 2).
        # Wait, the scaling is "scaling the set to diameter 2".
        # So we compute Delta = prod(d_ij) where d_ij are scaled distances.
        # Then score = Delta / n^n.
        # So log(score) = log(Delta) - n * log(n).
        return log_prod - n * np.log(n)

    def log_product_smooth(points, beta):
        """
        Smooth surrogate for the objective using soft-max for diameter.
        Returns -log(score) for minimization.
        """
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq)
        
        mask = ~np.eye(n, dtype=bool)
        off_diag_dists = dists[mask]
        
        if len(off_diag_dists) == 0:
            return np.inf
        
        # Soft-max for diameter
        scaled = off_diag_dists / beta
        m = np.max(scaled)
        log_sum_exp = m + np.log(np.sum(np.exp(scaled - m)))
        diameter = beta * log_sum_exp
        
        if diameter < 1e-15:
            return np.inf
        
        scaled_dists = off_diag_dists * (2.0 / diameter)
        scaled_dists = np.maximum(scaled_dists, 1e-15)
        
        log_prod = np.sum(np.log(scaled_dists))
        # We want to maximize log_prod - n*log(n)
        # So we minimize -(log_prod - n*log(n)) = n*log(n) - log_prod
        return n * np.log(n) - log_prod

    # Baseline: Regular Octagon
    angles = 2 * np.pi * np.arange(n) / n
    reg_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    best_score_val = exact_score(reg_oct) # Should be ~0 (log of 1)
    best_points = reg_oct.copy()
    
    # Helper to update best
    def try_update(points):
        nonlocal best_score_val, best_points
        if np.any(np.isnan(points)):
            return
        # Check distinct points
        d = np.sqrt(np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :])**2, axis=2))
        np.fill_diagonal(d, np.inf)
        if np.min(d) < 1e-8:
            return
        
        s = exact_score(points)
        if s > best_score_val:
            best_score_val = s
            best_points = points.copy()

    # Generate diverse initializations
    starts = [reg_oct.flatten()]

    # Two squares with various relative rotations
    for rot in np.linspace(0.0, 0.49, 20):
        a1 = 2 * np.pi * np.arange(4) / 4
        a2 = 2 * np.pi * (np.arange(4) / 4 + rot)
        sq1 = np.column_stack([np.cos(a1), np.sin(a1)])
        sq2 = np.column_stack([np.cos(a2), np.sin(a2)])
        starts.append(np.vstack([sq1, sq2]).flatten())

    # Antipodal pairs
    for seed_val in range(10):
        np.random.seed(seed_val)
        base_angles = np.sort(np.random.uniform(0, np.pi, 4))
        all_angles = np.concatenate([base_angles, base_angles + np.pi])
        pts = np.column_stack([np.cos(all_angles), np.sin(all_angles)])
        starts.append(pts.flatten())
        
    # Random points
    np.random.seed(123)
    for _ in range(30):
        pts = np.random.uniform(-1, 1, (n, 2))
        starts.append(pts.flatten())

    # Points on circle with perturbations
    for _ in range(20):
        base_angles = 2 * np.pi * np.arange(n) / n + np.random.uniform(-0.2, 0.2, n)
        radii = 1.0 + np.random.uniform(-0.2, 0.2, n)
        pts = np.column_stack([np.cos(base_angles) * radii, np.sin(base_angles) * radii])
        starts.append(pts.flatten())

    # NEW: Add more structured initializations
    # Configuration: 4 points on inner circle, 4 on outer circle (alternating)
    for r_inner in [0.5, 0.6, 0.7, 0.8]:
        for rot_offset in [0.0, 0.1, 0.2, 0.3]:
            a_inner = 2 * np.pi * np.arange(4) / 4 + rot_offset
            a_outer = 2 * np.pi * (np.arange(4) / 4 + rot_offset + 0.25)
            pts_in = np.column_stack([np.cos(a_inner) * r_inner, np.sin(a_inner) * r_inner])
            pts_out = np.column_stack([np.cos(a_outer), np.sin(a_outer)])
            starts.append(np.vstack([pts_in, pts_out]).flatten())

    # Configuration: 2 squares at different scales
    for scale in [0.6, 0.7, 0.8, 0.9]:
        for rot in [0.0, 0.1, 0.2, 0.3, 0.4]:
            a1 = 2 * np.pi * np.arange(4) / 4
            a2 = 2 * np.pi * (np.arange(4) / 4 + rot)
            sq1 = np.column_stack([np.cos(a1) * scale, np.sin(a1) * scale])
            sq2 = np.column_stack([np.cos(a2), np.sin(a2)])
            starts.append(np.vstack([sq1, sq2]).flatten())

    # Phase 1: Local optimization with soft-max surrogate
    # Use a range of betas to balance smoothness and accuracy
    betas = [10.0, 50.0, 100.0]
    
    for s in starts:
        if time.time() > deadline - 30:
            break
        for beta in betas:
            if time.time() > deadline - 30:
                break
            try:
                res = minimize(
                    lambda x, b=beta: log_product_smooth(x.reshape(n, 2), b),
                    s,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < np.inf:
                    candidate = res.x.reshape(n, 2)
                    try_update(candidate)
            except Exception:
                pass

    # Phase 2: Refine current best with higher beta
    if time.time() < deadline - 20:
        for beta_val in [200.0, 500.0, 1000.0, 5000.0]:
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

    # Phase 3: Global optimization with differential evolution
    if time.time() < deadline - 30:
        bounds = [(-5, 5)] * (2 * n)
        try:
            de_res = differential_evolution(
                lambda x: log_product_smooth(x, 100.0),
                bounds,
                seed=42,
                maxiter=200,
                popsize=20,
                tol=1e-8,
                polish=True,
                updateinterval=10
            )
            if de_res.fun < np.inf:
                candidate = de_res.x.reshape(n, 2)
                try_update(candidate)
        except Exception:
            pass

    # Phase 4: High-precision polish with very high beta (approximating exact max)
    if time.time() < deadline - 5:
        for method in ['L-BFGS-B']:
            try:
                final_res = minimize(
                    lambda x: log_product_smooth(x.reshape(n, 2), 1e6),
                    best_points.flatten(),
                    method=method,
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if final_res.fun < np.inf:
                    candidate = final_res.x.reshape(n, 2)
                    try_update(candidate)
            except Exception:
                pass

    # Phase 5: NEW - Jitter and re-optimize to escape local minima
    if time.time() < deadline - 10:
        np.random.seed(42)
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            # Add small random jitter to best points
            jitter = np.random.normal(0, 0.05, (n, 2))
            jittered = best_points + jitter
            try:
                res = minimize(
                    lambda x: log_product_smooth(x.reshape(n, 2), 500.0),
                    jittered.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-9}
                )
                if res.fun < np.inf:
                    candidate = res.x.reshape(n, 2)
                    try_update(candidate)
            except Exception:
                pass

    # Phase 6: NEW - Final ultra-precise polish
    if time.time() < deadline - 3:
        try:
            final_res = minimize(
                lambda x: log_product_smooth(x.reshape(n, 2), 1e8),
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if final_res.fun < np.inf:
                candidate = final_res.x.reshape(n, 2)
                try_update(candidate)
        except Exception:
            pass

    # Final validation and bounds check
    dists = np.sqrt(np.sum((best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-10:
        best_points = reg_oct

    max_coord = np.max(np.abs(best_points))
    if max_coord > 1e6:
        best_points = best_points / max_coord * 1e5

    return [[float(p[0]), float(p[1])] for p in best_points]
