"""
Strategy: Distance-space optimization + MDS embedding + coordinate polish.
1. Parameterize the 28 pairwise distances (log-space), constrain max=2 (diameter).
2. Maximize sum of log(distances) in distance space (smooth, differentiable).
3. Embed resulting distance matrix into 2D via classical MDS.
4. Polish in coordinate space with Nelder-Mead.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

_START_TIME = time.time()
_DEADLINE = 135.0  # Safety margin before 150s kill

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
    if time.time() > _DEADLINE:
        return 1e10
    pts = params.reshape(8, 2)
    for i in range(8):
        for j in range(i+1, 8):
            d = math.sqrt((pts[i][0]-pts[j][0])**2 + (pts[i][1]-pts[j][1])**2)
            if d < 1e-10:
                return 1e10
    ls = _log_score(pts)
    if not np.isfinite(ls):
        return 1e10
    return -ls

def _mds_2d(dmat):
    """Classical MDS: embed a distance matrix into 2D."""
    # dmat: 8x8 distance matrix
    n = dmat.shape[0]
    # Squared distances
    D2 = dmat**2
    # Double centering
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(B)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # Take top 2
    coords = eigvecs[:, :2] * np.sqrt(np.maximum(eigvals[:2], 0))
    return coords

def _log_distance_objective(log_dists):
    """
    Objective in log-distance space.
    log_dists: 28-vector of log(pairwise distances).
    We want to maximize sum(log_dists) subject to max(dist) <= 2 and dist > 0.
    Using soft constraint: penalty for dist > 2.
    """
    if time.time() > _DEADLINE:
        return 1e10
    dists = np.exp(log_dists)
    if np.any(dists <= 0) or np.any(~np.isfinite(dists)):
        return 1e10
    # Penalty for exceeding diameter 2
    excess = np.maximum(dists - 2.0, 0)
    penalty = 100.0 * np.sum(excess**2)
    # Also penalize very small distances
    small_penalty = 10.0 * np.sum(np.maximum(0, 0.01 - dists)**2)
    obj = -np.sum(log_dists) + penalty + small_penalty
    if not np.isfinite(obj):
        return 1e10
    return obj

def _log_distance_objective_v2(log_dists, beta=50.0):
    """
    Smoother version: use soft-max for the diameter constraint.
    """
    if time.time() > _DEADLINE:
        return 1e10
    dists = np.exp(log_dists)
    if np.any(dists <= 0) or np.any(~np.isfinite(dists)):
        return 1e10
    max_d = np.max(dists)
    shifted = beta * (dists - max_d)
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    smooth_max = max_d + log_sum_exp / beta
    # Objective: maximize sum(log_dists) - penalty * smooth_max
    obj = -np.sum(log_dists) + 50.0 * smooth_max
    # Small distance penalty
    small_penalty = 5.0 * np.sum(np.maximum(0, 0.01 - dists)**2)
    obj += small_penalty
    if not np.isfinite(obj):
        return 1e10
    return obj

def _regular_octagon():
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _octagon_distances():
    """Get the 28 pairwise distances of a regular octagon (radius 1, so diameter = 2*sin(3pi/4)=sqrt(2)*... actually 2*cos(pi/8) for radius 1)."""
    pts = _regular_octagon()
    # Scale to diameter 2
    d = pdist(pts)
    max_d = np.max(d)
    d_scaled = d / max_d * 2.0
    return d_scaled

def _random_distance_matrix(rng):
    """Generate a random set of 28 distances that might be embeddable in 2D."""
    # Start from octagon and perturb
    d0 = _octagon_distances()
    log_d = np.log(d0) + rng.normal(0, 0.3, 28)
    # Ensure max is around 2
    d = np.exp(log_d)
    d = d / np.max(d) * 2.0
    return d

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 135.0

    best_points = _regular_octagon()
    best_score = _score(best_points)
    best_log_score = _log_score(best_points)

    rng = np.random.default_rng(42)

    # --- Phase 1: Optimize in log-distance space ---
    # Initial log-distance vectors
    init_log_dists = []

    # From regular octagon
    d_oct = _octagon_distances()
    init_log_dists.append(np.log(d_oct))

    # Perturbed octagon distances
    for _ in range(40):
        d0 = _octagon_distances()
        log_d = np.log(d0) + rng.normal(0, 0.2, 28)
        d = np.exp(log_d)
        d = d / np.max(d) * 2.0
        init_log_dists.append(np.log(d))

    # Two-square configurations
    for r1 in [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]:
        for r2 in [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]:
            if abs(r1 - r2) < 0.15:
                continue
            for rot1 in [0.0, np.pi/8]:
                for rot2 in [0.0, np.pi/8]:
                    pts = []
                    for k in range(4):
                        a = rot1 + 2*np.pi*k/4
                        pts.append([r1*np.cos(a), r1*np.sin(a)])
                    for k in range(4):
                        a = rot2 + 2*np.pi*k/4
                        pts.append([r2*np.cos(a), r2*np.sin(a)])
                    pts = np.array(pts)
                    d = pdist(pts)
                    max_d = np.max(d)
                    if max_d > 0:
                        d = d / max_d * 2.0
                        if np.all(d > 0.01):
                            init_log_dists.append(np.log(d))

    # Antipodal pair configurations
    for _ in range(20):
        base_angles = rng.uniform(0, np.pi/2, 4)
        radii = rng.uniform(0.4, 1.5, 4)
        pts = []
        for i in range(4):
            a = base_angles[i]
            r = radii[i]
            pts.append([r*np.cos(a), r*np.sin(a)])
            pts.append([-r*np.cos(a), -r*np.sin(a)])
        pts = np.array(pts)
        d = pdist(pts)
        max_d = np.max(d)
        if max_d > 0:
            d = d / max_d * 2.0
            if np.all(d > 0.01):
                init_log_dists.append(np.log(d))

    # Random points
    for _ in range(30):
        radii = rng.uniform(0.1, 1.8, 8)
        angles_r = rng.uniform(0, 2*np.pi, 8)
        pts = np.column_stack([radii*np.cos(angles_r), radii*np.sin(angles_r)])
        d = pdist(pts)
        max_d = np.max(d)
        if max_d > 0:
            d = d / max_d * 2.0
            if np.all(d > 0.01):
                init_log_dists.append(np.log(d))

    # Score initializations by their log-sum (proxy for quality)
    scored_inits = []
    for ld in init_log_dists:
        if time.time() > _DEADLINE - 50:
            break
        d = np.exp(ld)
        if np.all(d > 0.005) and np.max(d) <= 2.5:
            scored_inits.append((np.sum(ld), ld))

    scored_inits.sort(key=lambda x: x[0], reverse=True)

    # Optimize in log-distance space
    best_log_dists = None
    best_ld_score = -np.inf

    for ls, ld0 in scored_inits[:20]:
        if time.time() > _DEADLINE - 30:
            break
        x0 = ld0.copy()
        try:
            res = minimize(_log_distance_objective_v2, x0, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < 1e10:
                d_opt = np.exp(res.x)
                # Check validity
                if np.all(d_opt > 0.005) and np.max(d_opt) < 2.1:
                    s = np.sum(np.log(d_opt))
                    if s > best_ld_score:
                        best_ld_score = s
                        best_log_dists = res.x.copy()
        except Exception:
            pass

    # Also try Nelder-Mead on distance space for top inits
    for ls, ld0 in scored_inits[:10]:
        if time.time() > _DEADLINE - 25:
            break
        x0 = ld0.copy()
        try:
            res = minimize(_log_distance_objective, x0, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            if res.fun < 1e10:
                d_opt = np.exp(res.x)
                if np.all(d_opt > 0.005) and np.max(d_opt) < 2.1:
                    s = np.sum(np.log(d_opt))
                    if s > best_ld_score:
                        best_ld_score = s
                        best_log_dists = res.x.copy()
        except Exception:
            pass

    # --- Phase 2: Embed optimized distances into 2D ---
    candidate_points_list = []

    # Embed the best distance matrix
    if best_log_dists is not None:
        d_mat = np.exp(best_log_dists)
        # Build full distance matrix
        dmat = squareform(d_mat)
        np.fill_diagonal(dmat, 0)
        # Ensure symmetry
        dmat = (dmat + dmat.T) / 2
        coords = _mds_2d(dmat)
        if np.all(np.isfinite(coords)) and np.all(np.abs(coords) < 1e5):
            candidate_points_list.append(coords)

    # Also embed from top-5 distance-space solutions
    # Re-run to get top solutions
    top_dists = []
    for ls, ld0 in scored_inits[:15]:
        if time.time() > _DEADLINE - 20:
            break
        x0 = ld0.copy()
        try:
            res = minimize(_log_distance_objective_v2, x0, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < 1e10:
                d_opt = np.exp(res.x)
                if np.all(d_opt > 0.005) and np.max(d_opt) < 2.1:
                    top_dists.append((np.sum(np.log(d_opt)), d_opt))
        except Exception:
            pass

    top_dists.sort(key=lambda x: x[0], reverse=True)
    for s, d_opt in top_dists[:5]:
        dmat = squareform(d_opt)
        np.fill_diagonal(dmat, 0)
        dmat = (dmat + dmat.T) / 2
        coords = _mds_2d(dmat)
        if np.all(np.isfinite(coords)) and np.all(np.abs(coords) < 1e5):
            candidate_points_list.append(coords)

    # Also add the regular octagon and some coordinate-space candidates
    candidate_points_list.append(_regular_octagon())

    # Add perturbed octagon
    for _ in range(10):
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False) + rng.normal(0, 0.2, 8)
        r = 1.0 + rng.normal(0, 0.15, 8)
        r = np.abs(r) + 0.1
        pts = np.column_stack([r*np.cos(angles), r*np.sin(angles)])
        candidate_points_list.append(pts)

    # --- Phase 3: Polish in coordinate space ---
    # Score all candidates
    scored_candidates = []
    for cp in candidate_points_list:
        if time.time() > _DEADLINE - 15:
            break
        try:
            ls = _log_score(cp)
            if np.isfinite(ls):
                scored_candidates.append((ls, cp))
        except Exception:
            pass

    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    # Update best
    if scored_candidates:
        ls_best, pts_best = scored_candidates[0]
        if ls_best > best_log_score:
            best_log_score = ls_best
            best_score = math.exp(ls_best) if ls_best < 50 else float('inf')
            best_points = pts_best

    # Nelder-Mead polish on top candidates
    for ls, cp in scored_candidates[:8]:
        if time.time() > _DEADLINE - 8:
            break
        x0 = cp.flatten()
        try:
            res = minimize(_neg_log_score, x0, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
            if res.fun < 1e10:
                opt_pts = res.x.reshape(8, 2)
                ls_val = _log_score(opt_pts)
                if np.isfinite(ls_val) and ls_val > best_log_score:
                    best_log_score = ls_val
                    best_score = math.exp(ls_val) if ls_val < 50 else float('inf')
                    best_points = opt_pts
        except Exception:
            pass

    # Final polish on the best
    if time.time() < _DEADLINE - 3:
        try:
            res = minimize(_neg_log_score, best_points.flatten(), method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < 1e10:
                opt_pts = res.x.reshape(8, 2)
                ls_val = _log_score(opt_pts)
                if np.isfinite(ls_val) and ls_val > best_log_score:
                    best_log_score = ls_val
                    best_score = math.exp(ls_val) if ls_val < 50 else float('inf')
                    best_points = opt_pts
        except Exception:
            pass

    # Ensure coordinates are within bounds
    final_result = []
    for p in best_points:
        x = max(-1e6, min(1e6, float(p[0])))
        y = max(-1e6, min(1e6, float(p[1])))
        final_result.append([x, y])

    return final_result
