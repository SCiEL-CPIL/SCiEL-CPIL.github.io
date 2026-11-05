import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s kill

    n = 8

    def exact_log_score(points):
        """Compute log of score: log(prod of ordered-pair distances after scaling to diameter 2) - n*log(n)."""
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq)
        mask = ~np.eye(n, dtype=bool)
        off_diag = dists[mask]
        if len(off_diag) == 0 or np.any(np.isnan(off_diag)):
            return -np.inf
        diameter = np.max(off_diag)
        if diameter < 1e-15:
            return -np.inf
        scaled = off_diag * (2.0 / diameter)
        scaled = np.maximum(scaled, 1e-15)
        log_prod = np.sum(np.log(scaled))
        return log_prod - n * np.log(n)

    def softmax_log_score_neg(x, beta):
        """Negative log-score with soft-max diameter surrogate for minimization."""
        p = x.reshape(n, 2)
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq)
        mask = ~np.eye(n, dtype=bool)
        off_diag = dists[mask]
        if len(off_diag) == 0:
            return np.inf
        # Soft-max approximation of max
        scaled_d = off_diag / beta
        m = np.max(scaled_d)
        log_sum_exp = m + np.log(np.sum(np.exp(scaled_d - m)))
        diameter_soft = beta * log_sum_exp
        if diameter_soft < 1e-15:
            return np.inf
        scaled_dists = off_diag * (2.0 / diameter_soft)
        scaled_dists = np.maximum(scaled_dists, 1e-15)
        log_prod = np.sum(np.log(scaled_dists))
        return n * np.log(n) - log_prod

    # Baseline: Regular Octagon
    angles = 2 * np.pi * np.arange(n) / n
    reg_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    best_log_score = exact_log_score(reg_oct)
    best_points = reg_oct.copy()

    def try_update(points):
        nonlocal best_log_score, best_points
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return
        # Check distinctness
        d = np.sqrt(np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :])**2, axis=2))
        np.fill_diagonal(d, np.inf)
        if np.min(d) < 1e-8:
            return
        s = exact_log_score(points)
        if s > best_log_score:
            best_log_score = s
            best_points = points.copy()

    # Generate diverse initializations
    starts = [reg_oct.flatten()]

    # Two squares with various relative rotations and radius ratios
    for rot in np.linspace(0.0, 0.49, 15):
        for r2 in [0.5, 0.7, 0.85, 1.0]:
            a1 = 2 * np.pi * np.arange(4) / 4
            a2 = 2 * np.pi * (np.arange(4) / 4 + rot)
            sq1 = np.column_stack([np.cos(a1), np.sin(a1)])
            sq2 = np.column_stack([np.cos(a2) * r2, np.sin(a2) * r2])
            starts.append(np.vstack([sq1, sq2]).flatten())

    # Antipodal pairs with varied angles
    for seed_val in range(8):
        np.random.seed(seed_val)
        base_angles = np.sort(np.random.uniform(0, np.pi, 4))
        all_angles = np.concatenate([base_angles, base_angles + np.pi])
        pts = np.column_stack([np.cos(all_angles), np.sin(all_angles)])
        starts.append(pts.flatten())

    # Random points
    np.random.seed(123)
    for _ in range(20):
        pts = np.random.uniform(-1.5, 1.5, (n, 2))
        starts.append(pts.flatten())

    # Perturbed regular octagon
    for _ in range(15):
        base_angles = 2 * np.pi * np.arange(n) / n + np.random.uniform(-0.15, 0.15, n)
        radii = 1.0 + np.random.uniform(-0.15, 0.15, n)
        pts = np.column_stack([np.cos(base_angles) * radii, np.sin(base_angles) * radii])
        starts.append(pts.flatten())

    # Phase 1: Local optimization with moderate beta
    betas_phase1 = [20.0, 50.0]
    for s in starts:
        if time.time() > deadline - 40:
            break
        for beta in betas_phase1:
            if time.time() > deadline - 40:
                break
            try:
                res = minimize(
                    lambda x, b=beta: softmax_log_score_neg(x, b),
                    s,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < np.inf:
                    try_update(res.x.reshape(n, 2))
            except Exception:
                pass

    # Phase 2: Refine best with higher beta
    if time.time() < deadline - 30:
        for beta_val in [100.0, 300.0, 1000.0]:
            if time.time() > deadline - 25:
                break
            try:
                res = minimize(
                    lambda x: softmax_log_score_neg(x, beta_val),
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if res.fun < np.inf:
                    try_update(res.x.reshape(n, 2))
            except Exception:
                pass

    # Phase 3: Gradient-based polish with exact normalization
    if time.time() < deadline - 10:
        p = best_points.copy()
        step_size = 0.005
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=2)
            mask = ~np.eye(n, dtype=bool)
            dists_sq_m = dists_sq[mask].copy()
            dists_sq_m = np.maximum(dists_sq_m, 1e-12)
            inv_dists_sq = np.zeros_like(dists_sq)
            inv_dists_sq[mask] = 1.0 / dists_sq_m
            grad = np.sum(diff * inv_dists_sq[:, :, np.newaxis], axis=1)
            new_pts = p + step_size * grad
            # Normalize to diameter 2
            diff_new = new_pts[:, np.newaxis, :] - new_pts[np.newaxis, :, :]
            dists_new = np.sqrt(np.sum(diff_new**2, axis=2))
            D = np.max(dists_new)
            if D < 1e-10:
                break
            new_pts = new_pts * (2.0 / D)
            try_update(new_pts)
            if np.allclose(new_pts, p, atol=1e-10):
                break
            p = new_pts

    # Final validation
    dists = np.sqrt(np.sum((best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-10:
        best_points = reg_oct

    max_coord = np.max(np.abs(best_points))
    if max_coord >= 1e6:
        best_points = best_points * (1e6 / max_coord)

    return best_points.tolist()
