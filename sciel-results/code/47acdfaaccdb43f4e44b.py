import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    n = 8

    def compute_score(pts):
        """Calculate the actual score: Delta / n^n where Delta is product of pairwise distances after scaling to diameter 2."""
        pts = np.array(pts, dtype=np.float64)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq)
        iu = np.triu_indices(n, 1)
        d = dists[iu]
        D = np.max(d)
        if D < 1e-9:
            return -np.inf
        scale = 2.0 / D
        d_scaled = d * scale
        d_scaled = np.maximum(d_scaled, 1e-15)
        log_delta = np.sum(np.log(d_scaled))
        delta = np.exp(log_delta)
        score = delta / (n ** n)
        return score

    def two_squares(r1, r2, theta1, theta2):
        """Generate 8 points as two concentric squares with independent radii and rotations."""
        angles1 = theta1 + np.pi / 2 * np.arange(4)
        pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        angles2 = theta2 + np.pi / 2 * np.arange(4)
        pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        return np.vstack([pts1, pts2])

    def neg_log_score_4d(params):
        """Negative of log(Delta) - 28*log(D) for the 4D two-squares parameterization."""
        if time.time() > deadline:
            return 1e10
        r1, r2, theta1, theta2 = params
        if r1 < 1e-6 or r2 < 1e-6:
            return 1e10
        pts = two_squares(r1, r2, theta1, theta2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq)
        iu = np.triu_indices(n, 1)
        d = dists[iu]
        if np.min(d) < 1e-8:
            return 1e10
        D = np.max(d)
        if D < 1e-9:
            return 1e10
        log_prod = np.sum(np.log(d))
        return 28.0 * np.log(D) - log_prod

    def neg_log_score_16d(pts_flat):
        """Negative of log(Delta) - 28*log(D) for full 16D point coordinates."""
        if time.time() > deadline:
            return 1e10
        pts = pts_flat.reshape(n, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq)
        iu = np.triu_indices(n, 1)
        d = dists[iu]
        if np.min(d) < 1e-8:
            return 1e10
        D = np.max(d)
        if D < 1e-9:
            return 1e10
        log_prod = np.sum(np.log(d))
        return 28.0 * np.log(D) - log_prod

    best_score = -np.inf
    best_pts = None

    # Phase 1: Comprehensive grid search over the 4D two-squares manifold
    # By symmetry, theta can be restricted to [0, pi/4) due to square symmetry
    # but we'll use [0, pi/2) to be safe and let the optimizer find the best
    
    # Coarse grid
    n_r = 15
    n_t = 12
    
    r1_vals = np.linspace(0.3, 2.5, n_r)
    r2_vals = np.linspace(0.1, 2.0, n_r)
    theta_vals = np.linspace(0.0, np.pi/2, n_t)
    
    for r1 in r1_vals:
        if time.time() > deadline - 40:
            break
        for r2 in r2_vals:
            for t1 in theta_vals:
                if time.time() > deadline - 35:
                    break
                for t2 in theta_vals:
                    params = np.array([r1, r2, t1, t2])
                    obj = neg_log_score_4d(params)
                    if obj < 1e9:
                        pts = two_squares(r1, r2, t1, t2)
                        sc = compute_score(pts)
                        if sc > best_score:
                            best_score = sc
                            best_pts = pts.copy()
                            best_params = params.copy()
    
    # Include regular octagon as baseline
    angles_oct = 2 * np.pi * np.arange(8) / 8
    octagon = np.column_stack([np.cos(angles_oct), np.sin(angles_oct)])
    oct_score = compute_score(octagon)
    if oct_score > best_score:
        best_score = oct_score
        best_pts = octagon.copy()
        # For the octagon, r1=r2=1, theta1=0, theta2=pi/4
        best_params = np.array([1.0, 1.0, 0.0, np.pi/4])
    else:
        if best_pts is None:
            best_pts = octagon.copy()
            best_score = oct_score
            best_params = np.array([1.0, 1.0, 0.0, np.pi/4])

    # Phase 2: Local refinement in 4D parameter space
    if time.time() < deadline - 25:
        bounds_4d = [(0.05, 5.0), (0.05, 5.0), (0.0, np.pi), (0.0, np.pi)]
        
        try:
            res = minimize(
                neg_log_score_4d,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if res.fun < 1e9:
                cand_pts = two_squares(res.x[0], res.x[1], res.x[2], res.x[3])
                cand_score = compute_score(cand_pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = cand_pts
                    best_params = res.x.copy()
        except Exception:
            pass
        
        try:
            res2 = minimize(
                neg_log_score_4d,
                best_params,
                method='L-BFGS-B',
                bounds=bounds_4d,
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res2.fun < 1e9:
                cand_pts = two_squares(res2.x[0], res2.x[1], res2.x[2], res2.x[3])
                cand_score = compute_score(cand_pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = cand_pts
                    best_params = res2.x.copy()
        except Exception:
            pass

    # Phase 3: Full 16D local refinement to break symmetry if beneficial
    if time.time() < deadline - 15:
        best_flat = best_pts.flatten()
        
        try:
            res3 = minimize(
                neg_log_score_16d,
                best_flat,
                method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if res3.fun < 1e9:
                cand_pts = res3.x.reshape(n, 2)
                cand_score = compute_score(cand_pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = cand_pts
        except Exception:
            pass
        
        try:
            res4 = minimize(
                neg_log_score_16d,
                best_flat,
                method='L-BFGS-B',
                options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res4.fun < 1e9:
                cand_pts = res4.x.reshape(n, 2)
                cand_score = compute_score(cand_pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = cand_pts
        except Exception:
            pass

    # Phase 4: Additional 4D refinements from multiple starting points if time permits
    if time.time() < deadline - 10:
        # Try a few different initial conditions for 4D optimization
        alt_starts = [
            np.array([1.0, 0.8, 0.0, np.pi/4]),
            np.array([1.0, 1.2, 0.0, np.pi/4]),
            np.array([1.2, 0.8, 0.0, np.pi/4]),
            np.array([0.8, 1.2, 0.0, np.pi/4]),
            np.array([1.0, 1.0, np.pi/8, np.pi/8 + np.pi/4]),
        ]
        
        for start in alt_starts:
            if time.time() > deadline - 8:
                break
            try:
                res_alt = minimize(
                    neg_log_score_4d,
                    start,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12}
                )
                if res_alt.fun < 1e9:
                    cand_pts = two_squares(res_alt.x[0], res_alt.x[1], res_alt.x[2], res_alt.x[3])
                    cand_score = compute_score(cand_pts)
                    if cand_score > best_score:
                        best_score = cand_score
                        best_pts = cand_pts
            except Exception:
                pass

    # Phase 5: Final 16D polish
    if time.time() < deadline - 3:
        best_flat = best_pts.flatten()
        try:
            res_final = minimize(
                neg_log_score_16d,
                best_flat,
                method='Nelder-Mead',
                options={'maxiter': 30000, 'xatol': 1e-14, 'fatol': 1e-14}
            )
            if res_final.fun < 1e9:
                cand_pts = res_final.x.reshape(n, 2)
                cand_score = compute_score(cand_pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = cand_pts
        except Exception:
            pass

    # Final validation
    if best_pts is None:
        angles = 2 * np.pi * np.arange(n) / n
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Ensure distinct points
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-8:
        angles = 2 * np.pi * np.arange(n) / n
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Ensure coordinates are within bounds
    max_coord = np.max(np.abs(best_pts))
    if max_coord > 1e6:
        scale = 1e6 / max_coord
        best_pts = best_pts * scale

    return [[float(p[0]), float(p[1])] for p in best_pts]
