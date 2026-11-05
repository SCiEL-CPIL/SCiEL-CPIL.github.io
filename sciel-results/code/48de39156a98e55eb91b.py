"""
Strategy: Distance-space optimization + MDS embedding + differential evolution polish.
1. Parameterize 28 pairwise distances in log-space.
2. Maximize sum(log d_ij) - penalty * logsumexp(beta * (d_ij - 2)) in distance space.
   This is smooth and avoids the non-differentiability of max() in coordinate space.
3. Embed the resulting distance matrix into 2D via classical MDS.
4. Polish with differential evolution (global) + Nelder-Mead (local) in coordinate space.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

_START_TIME = time.time()
_DEADLINE = 135.0

def _log_score(points):
    """Compute log of Delta/n^n where Delta is product of distances over ordered pairs, after scaling to diameter 2."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    mask = ~np.eye(n, dtype=bool)
    valid_dists = dists[mask]
    if np.any(valid_dists <= 0):
        return -np.inf
    diameter = np.max(dists)
    if diameter <= 0:
        return -np.inf
    n_pairs = n * (n - 1)
    log_delta = n_pairs * np.log(2.0) + np.sum(np.log(valid_dists)) - n_pairs * np.log(diameter)
    log_score = log_delta - n * np.log(n)
    return log_score

def _score(points):
    ls = _log_score(points)
    if not np.isfinite(ls):
        return -np.inf
    if ls > 50:
        return float('inf')
    return math.exp(ls)

def _neg_log_score(params):
    """Negative log score for coordinate-space minimization."""
    if time.time() > _DEADLINE:
        return 1e10
    pts = params.reshape(8, 2)
    for i in range(8):
        for j in range(i + 1, 8):
            d = math.sqrt((pts[i][0] - pts[j][0])**2 + (pts[i][1] - pts[j][1])**2)
            if d < 1e-10:
                return 1e10
    ls = _log_score(pts)
    if not np.isfinite(ls):
        return 1e10
    return -ls

def _neg_log_score_coords(params):
    """Same as _neg_log_score but with deadline check."""
    return _neg_log_score(params)

def _neg_dist_space(log_dists, beta=20.0, penalty_weight=1.0):
    """
    Negative objective in distance space.
    log_dists: 28-element array of log(d_ij) for i<j.
    We want to maximize sum(log_dists) subject to max(d_ij) <= 2.
    Penalty: penalty_weight * logsumexp(beta * (d_ij - 2))
    """
    if time.time() > _DEADLINE:
        return 1e10
    d = np.exp(log_dists)
    if np.any(d <= 0) or np.any(np.isnan(d)) or np.any(np.isinf(d)):
        return 1e10
    # Smooth diameter penalty
    shifted = beta * (d - 2.0)
    # Clip for numerical stability
    shifted = np.clip(shifted, -500, 500)
    max_shifted = np.max(shifted)
    log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    penalty = penalty_weight * log_sum_exp
    objective = -np.sum(log_dists) + penalty
    if not np.isfinite(objective):
        return 1e10
    return objective

def _dist_matrix_from_log(log_dists):
    """Reconstruct 8x8 distance matrix from 28 log-distances (i<j)."""
    n = 8
    D = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = np.exp(log_dists[idx])
            D[j, i] = np.exp(log_dists[idx])
            idx += 1
    return D

def _mds_2d(D):
    """Classical MDS: embed distance matrix D into 2D."""
    n = D.shape[0]
    # Double centering
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(B)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # Take top 2
    if eigvals[0] < 0 or eigvals[1] < 0:
        # Fallback: return regular octagon
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])
    X = eigvecs[:, :2] * np.sqrt(np.maximum(eigvals[:2], 0))
    return X

def _regular_octagon():
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _dist_from_points(pts):
    """Extract 28 pairwise distances (i<j) from 8 points."""
    n = 8
    d = np.zeros(28)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            d[idx] = np.sqrt(np.sum((pts[i] - pts[j])**2))
            idx += 1
    return d

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 135.0

    best_points = _regular_octagon()
    best_log_score = _log_score(best_points)

    rng = np.random.default_rng(123)

    # === Stage 1: Distance-space optimization ===
    # Initial log-distances from regular octagon
    oct_pts = _regular_octagon()
    oct_dists = _dist_from_points(oct_pts)
    oct_dists = np.maximum(oct_dists, 1e-6)
    init_log_dists = np.log(oct_dists)

    # Also try other initial distance configurations
    init_configs_dist = [init_log_dists.copy()]

    # Two concentric squares distance configs
    for r1 in [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]:
        for r2 in [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]:
            if abs(r1 - r2) < 0.15:
                continue
            for rot_offset in [0.0, np.pi/8, np.pi/4]:
                pts = []
                for k in range(4):
                    a = rot_offset + 2 * np.pi * k / 4
                    pts.append([r1 * np.cos(a), r1 * np.sin(a)])
                for k in range(4):
                    a = rot_offset + np.pi/4 + 2 * np.pi * k / 4
                    pts.append([r2 * np.cos(a), r2 * np.sin(a)])
                pts = np.array(pts)
                d = _dist_from_points(pts)
                d = np.maximum(d, 1e-6)
                init_configs_dist.append(np.log(d))

    # Random distance configs
    for _ in range(20):
        d = rng.uniform(0.2, 2.0, 28)
        init_configs_dist.append(np.log(d))

    # Optimize in distance space using L-BFGS-B
    best_log_dists = None
    best_dist_obj = np.inf

    for init_ld in init_configs_dist:
        if time.time() > _DEADLINE - 60:
            break
        for beta in [15.0, 30.0, 60.0]:
            if time.time() > _DEADLINE - 55:
                break
            try:
                res = minimize(
                    lambda x, b=beta: _neg_dist_space(x, beta=b, penalty_weight=1.0),
                    init_ld,
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.fun < 1e10 and res.fun < best_dist_obj:
                    best_dist_obj = res.fun
                    best_log_dists = res.x.copy()
            except Exception:
                pass

    # Also try Nelder-Mead in distance space for top candidates
    if best_log_dists is not None and time.time() < _DEADLINE - 50:
        try:
            res = minimize(
                lambda x: _neg_dist_space(x, beta=30.0, penalty_weight=1.0),
                best_log_dists,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if res.fun < 1e10 and res.fun < best_dist_obj:
                best_dist_obj = res.fun
                best_log_dists = res.x.copy()
        except Exception:
            pass

    # === Stage 2: MDS embedding ===
    candidate_points_list = []

    if best_log_dists is not None:
        D = _dist_matrix_from_log(best_log_dists)
        # Ensure triangle inequality is roughly satisfied; MDS handles this
        X = _mds_2d(D)
        candidate_points_list.append(X)

    # Also embed from a few other good distance configs
    for init_ld in init_configs_dist[:5]:
        if time.time() > _DEADLINE - 45:
            break
        D = _dist_matrix_from_log(init_ld)
        X = _mds_2d(D)
        candidate_points_list.append(X)

    # Add regular octagon and random points
    candidate_points_list.append(_regular_octagon())
    for _ in range(10):
        radii = rng.uniform(0.3, 1.8, 8)
        angles = rng.uniform(0, 2 * np.pi, 8)
        candidate_points_list.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]))

    # Antipodal pairs
    for _ in range(8):
        base_angles = rng.uniform(0, np.pi/2, 4)
        radii = rng.uniform(0.5, 1.5, 4)
        pts = []
        for i in range(4):
            a = base_angles[i]
            r = radii[i]
            pts.append([r * np.cos(a), r * np.sin(a)])
            pts.append([-r * np.cos(a), -r * np.sin(a)])
        candidate_points_list.append(np.array(pts))

    # Score and sort
    scored_candidates = []
    for cand in candidate_points_list:
        if time.time() > _DEADLINE - 40:
            break
        try:
            ls = _log_score(cand)
            if np.isfinite(ls):
                scored_candidates.append((ls, cand))
        except Exception:
            pass

    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    # Update best if any candidate is better
    for ls, cand in scored_candidates:
        if ls > best_log_score:
            best_log_score = ls
            best_points = cand

    # === Stage 3: Coordinate-space polish ===
    # 3a: Differential evolution (global) on top candidates
    if time.time() < _DEADLINE - 25:
        for ls, cand in scored_candidates[:5]:
            if time.time() > _DEADLINE - 20:
                break
            x0 = cand.flatten()
            # Use DE with bounds around the candidate
            bounds = [(x - 0.5, x + 0.5) for x in x0]
            # Also allow wider search
            bounds = [(-2.5, 2.5)] * 16
            try:
                de_result = differential_evolution(
                    _neg_log_score_coords,
                    bounds,
                    seed=42,
                    maxiter=200,
                    popsize=15,
                    tol=1e-8,
                    polish=True,
                    init='sohs'
                )
                if de_result.fun < 1e10:
                    opt_pts = de_result.x.reshape(8, 2)
                    ls_val = _log_score(opt_pts)
                    if np.isfinite(ls_val) and ls_val > best_log_score:
                        best_log_score = ls_val
                        best_points = opt_pts
            except Exception:
                pass

    # 3b: Nelder-Mead polish on top candidates
    if time.time() < _DEADLINE - 15:
        for ls, cand in scored_candidates[:10]:
            if time.time() > _DEADLINE - 10:
                break
            x0 = cand.flatten()
            try:
                res = minimize(_neg_log_score, x0, method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
                if res.fun < 1e10:
                    opt_pts = res.x.reshape(8, 2)
                    ls_val = _log_score(opt_pts)
                    if np.isfinite(ls_val) and ls_val > best_log_score:
                        best_log_score = ls_val
                        best_points = opt_pts
            except Exception:
                pass

    # 3c: Smooth coordinate-space optimization with increasing beta (polish)
    if time.time() < _DEADLINE - 8:
        def _neg_smooth_coord(params, beta):
            if time.time() > _DEADLINE:
                return 1e10
            pts = params.reshape(8, 2)
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            mask = ~np.eye(8, dtype=bool)
            valid_dists = dists[mask]
            if np.any(valid_dists <= 0):
                return 1e10
            max_d = np.max(valid_dists)
            if max_d <= 0:
                return 1e10
            shifted = beta * (valid_dists - max_d)
            shifted = np.clip(shifted, -500, 500)
            max_s = np.max(shifted)
            log_sum_exp = max_s + np.log(np.sum(np.exp(shifted - max_s)))
            smooth_log_diam = max_d + log_sum_exp / beta
            n_pairs = 56
            log_delta = n_pairs * np.log(2.0) + np.sum(np.log(valid_dists)) - n_pairs * smooth_log_diam
            log_score = log_delta - 8 * np.log(8)
            if not np.isfinite(log_score):
                return 1e10
            return -log_score

        current = best_points.copy()
        for beta in [50.0, 200.0, 500.0, 1000.0]:
            if time.time() > _DEADLINE - 5:
                break
            try:
                res = minimize(lambda x, b=beta: _neg_smooth_coord(x, b), current.flatten(),
                               method='L-BFGS-B',
                               options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
                if res.fun < 1e10:
                    refined = res.x.reshape(8, 2)
                    ls_val = _log_score(refined)
                    if np.isfinite(ls_val) and ls_val > best_log_score:
                        best_log_score = ls_val
                        best_points = refined
                        current = refined.copy()
            except Exception:
                pass

    # 3d: Final Nelder-Mead polish
    if time.time() < _DEADLINE - 3:
        try:
            res = minimize(_neg_log_score, best_points.flatten(), method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < 1e10:
                refined = res.x.reshape(8, 2)
                ls_val = _log_score(refined)
                if np.isfinite(ls_val) and ls_val > best_log_score:
                    best_log_score = ls_val
                    best_points = refined
        except Exception:
            pass

    # Ensure coordinates are within bounds
    final_result = []
    for p in best_points:
        x = max(-1e6, min(1e6, float(p[0])))
        y = max(-1e6, min(1e6, float(p[1])))
        final_result.append([x, y])

    return final_result
