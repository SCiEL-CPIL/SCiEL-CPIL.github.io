"""
Strategy: Distance-space optimization via MDS embedding.
- Represent the 8 points implicitly by their 28 pairwise log-distances.
- The diameter constraint becomes: max(log_dists) = 0 (after normalization).
- Objective: sum(log_dists) - 8*log(8) (since log_delta = 56*log(2) + sum(log_dists) - 56*log(diam),
  and after scaling diam=2, log_delta = sum(log_dists) + 56*log(2) - 56*log(2) = sum(log_dists)... 
  wait, let me be careful.
  
  Actually: score = Delta / n^n where Delta = product of all ordered-pair distances after scaling to diameter 2.
  Ordered pairs = n*(n-1) = 56.
  After scaling to diameter 2: all distances d_i become d_i / D * 2 where D is original diameter.
  So Delta = prod_{i != j} (d_ij / D * 2) = 2^56 * prod(d_ij) / D^56.
  log(Delta) = 56*log(2) + sum_{i!=j} log(d_ij) - 56*log(D).
  Score = Delta / 8^8, so log(score) = 56*log(2) + sum_{i!=j} log(d_ij) - 56*log(D) - 8*log(8).
  
  If we work in log-space with log_dists normalized so max = 0 (i.e., we set D=2 by scaling),
  then sum_{i!=j} log(d_ij/2) = sum_{i!=j} log(d_ij) - 56*log(2).
  And D=2, so -56*log(D) = -56*log(2).
  log(score) = 56*log(2) + [sum_{i!=j} log(d_ij) - 56*log(2)] - 56*log(2) - 8*log(8)
  = sum_{i!=j} log(d_ij) - 56*log(2) - 8*log(8).
  
  Hmm, let me just compute directly from the embedded points.
"""
import time
import math
import numpy as np
from scipy.optimize import differential_evolution, minimize, dual_annealing

_START_TIME = time.time()
_DEADLINE = 135.0


def _log_score(points):
    """Compute log of Delta/n^n where Delta is product of distances over ordered pairs,
    after scaling to diameter 2."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
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


def _neg_log_score(params):
    """Negative log score for coordinate-space minimization."""
    if time.time() > _DEADLINE:
        return 1e10
    pts = params.reshape(8, 2)
    for i in range(8):
        for j in range(i + 1, 8):
            d = math.sqrt((pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2)
            if d < 1e-10:
                return 1e10
    ls = _log_score(pts)
    if not np.isfinite(ls):
        return 1e10
    return -ls


def _regular_octagon():
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _mds_2d(dist_matrix):
    """Classical MDS: embed a 8x8 distance matrix into 2D."""
    n = dist_matrix.shape[0]
    # Squared distances
    D2 = dist_matrix ** 2
    # Double centering
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    # Eigen decomposition
    try:
        eigvals, eigvecs = np.linalg.eigh(B)
    except Exception:
        return None
    # Take top 2 eigenvalues (largest)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    if eigvals[0] < 0 or eigvals[1] < 0:
        # Negative eigenvalues mean the configuration isn't embeddable in 2D
        # Clamp to 0 for a best-effort embedding
        eigvals = np.maximum(eigvals, 0)
    pts = eigvecs[:, :2] * np.sqrt(eigvals[:2])
    return pts


def _dist_space_to_points(log_dists_28):
    """Convert 28 log-distances to 8 points via MDS."""
    # Reconstruct 8x8 distance matrix
    D = np.zeros((8, 8))
    idx = 0
    for i in range(8):
        for j in range(i + 1, 8):
            d = math.exp(log_dists_28[idx])
            D[i, j] = d
            D[j, i] = d
            idx += 1
    pts = _mds_2d(D)
    if pts is None:
        return None
    # Check distinctness
    for i in range(8):
        for j in range(i + 1, 8):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-10:
                return None
    return pts


def _dist_space_objective(log_dists_28):
    """
    Objective in distance space: minimize -log(score).
    log_dists_28: 28-element array of log-distances for pairs (0,1),(0,2),...,(6,7).
    We normalize so max log_dist = 0 (diameter = 2 after exp).
    """
    if time.time() > _DEADLINE:
        return 1e10
    
    # Normalize: shift so max = 0
    max_ld = np.max(log_dists_28)
    if max_ld > 50:  # Overflow protection
        return 1e10
    norm_ld = log_dists_28 - max_ld  # Now max is 0, all <= 0
    
    # All distances must be positive (log > -inf), which is guaranteed
    # But we need the MDS embedding to work
    pts = _dist_space_to_points(norm_ld)
    if pts is None:
        return 1e10
    
    # Check that the embedded points actually have a reasonable diameter
    # (MDS might not perfectly preserve distances)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    mask = ~np.eye(8, dtype=bool)
    valid_dists = dists[mask]
    if np.any(valid_dists <= 1e-10):
        return 1e10
    
    # Compute actual score from embedded points
    ls = _log_score(pts)
    if not np.isfinite(ls):
        return 1e10
    
    # Penalty: if the MDS embedding doesn't preserve the distance structure well,
    # the actual diameter of embedded points may differ from 2.
    # But _log_score already handles scaling, so we're fine.
    
    return -ls


def _neg_smooth_coord(params, beta):
    """Smooth coordinate-space objective with log-sum-exp diameter."""
    if time.time() > _DEADLINE:
        return 1e10
    pts = params.reshape(8, 2)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
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


def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 135.0

    best_points = _regular_octagon()
    best_log_score = _log_score(best_points)

    rng = np.random.default_rng(42)

    # === Stage 1: Distance-space optimization via differential evolution ===
    # 28 log-distances. Bounds: log-dists should be in reasonable range.
    # After normalization (max=0), values are <= 0. 
    # For the search, let's use raw log-dists with bounds [-8, 2] (distances from ~0.0003 to ~7.4)
    
    dist_bounds = [(-8.0, 2.0)] * 28

    # Initial population for DE: include regular octagon distances
    oct_pts = _regular_octagon()
    oct_diff = oct_pts[:, np.newaxis, :] - oct_pts[np.newaxis, :]
    oct_dists = np.sqrt(np.sum(oct_diff ** 2, axis=2))
    oct_mask = ~np.eye(8, dtype=bool)
    oct_valid_dists = oct_dists[oct_mask]
    oct_log_dists = np.log(oct_valid_dists)
    
    # DE with multiple restarts
    de_best_obj = np.inf
    de_best_params = None
    
    n_restarts = 3
    for restart in range(n_restarts):
        if time.time() > _DEADLINE - 50:
            break
        try:
            result = differential_evolution(
                _dist_space_objective,
                bounds=dist_bounds,
                seed=42 + restart,
                maxiter=200,
                popsize=15,
                tol=1e-8,
                mutation=(0.3, 1.0),
                recombination=0.7,
                init='latinhypercube' if restart == 0 else 'sobol',
                polish=True,
                update_interval=1,
                callback=None
            )
            if result.fun < 1e10 and result.fun < de_best_obj:
                de_best_obj = result.fun
                de_best_params = result.x.copy()
        except Exception:
            pass

    # Convert DE result to points
    candidate_points_list = []
    if de_best_params is not None:
        pts_from_dist = _dist_space_to_points(de_best_params)
        if pts_from_dist is not None:
            candidate_points_list.append(pts_from_dist)

    # === Stage 2: Also try dual_annealing in distance space ===
    if time.time() < _DEADLINE - 45:
        try:
            da_result = dual_annealing(
                _dist_space_objective,
                x0=oct_log_dists,
                bounds=dist_bounds,
                maxiter=150,
                seed=42,
                initial_temp=3.0,
                restart_temp_ratio=2e-5,
                visit=[1.0, 1.0, 1.0, 1.0],
                accept=[1.0, 1.0, 1.0, 1.0],
                maxiter_no_improvement=30,
                polish=True,
                vectorized=False
            )
            if da_result.fun < 1e10:
                pts_from_da = _dist_space_to_points(da_result.x)
                if pts_from_da is not None:
                    candidate_points_list.append(pts_from_da)
        except Exception:
            pass

    # === Stage 3: Add diverse initial candidates in coordinate space ===
    candidate_points_list.append(_regular_octagon())
    
    # Two concentric squares with various parameters
    for r1 in [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]:
        for r2 in [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]:
            if abs(r1 - r2) < 0.3:
                continue
            for rot in [0.0, np.pi / 8, np.pi / 4]:
                pts = []
                for k in range(4):
                    a = rot + 2 * np.pi * k / 4
                    pts.append([r1 * np.cos(a), r1 * np.sin(a)])
                for k in range(4):
                    a = rot + np.pi / 4 + 2 * np.pi * k / 4
                    pts.append([r2 * np.cos(a), r2 * np.sin(a)])
                candidate_points_list.append(np.array(pts))

    # Antipodal pairs
    for _ in range(10):
        base_angles = rng.uniform(0, np.pi / 2, 4)
        radii = rng.uniform(0.5, 1.5, 4)
        pts = []
        for i in range(4):
            a = base_angles[i]
            r = radii[i]
            pts.append([r * np.cos(a), r * np.sin(a)])
            pts.append([-r * np.cos(a), -r * np.sin(a)])
        candidate_points_list.append(np.array(pts))

    # Random points
    for _ in range(15):
        radii = rng.uniform(0.3, 1.8, 8)
        angles = rng.uniform(0, 2 * np.pi, 8)
        candidate_points_list.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]))

    # Score and sort
    scored_candidates = []
    for cand in candidate_points_list:
        if time.time() > _DEADLINE - 30:
            break
        try:
            ls = _log_score(cand)
            if np.isfinite(ls):
                scored_candidates.append((ls, cand))
        except Exception:
            pass

    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    # Update best
    for ls, cand in scored_candidates:
        if ls > best_log_score:
            best_log_score = ls
            best_points = cand

    # === Stage 4: Coordinate-space polish on top candidates ===
    # 4a: Nelder-Mead on top candidates
    if time.time() < _DEADLINE - 15:
        for ls, cand in scored_candidates[:12]:
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

    # 4b: Smooth coordinate-space optimization with increasing beta
    if time.time() < _DEADLINE - 8:
        current = best_points.copy()
        for beta in [50.0, 200.0, 500.0, 1000.0, 2000.0]:
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

    # 4c: Final Nelder-Mead polish
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

    # 4d: One more round of DE in coordinate space from best point
    if time.time() < _DEADLINE - 5:
        try:
            coord_bounds = [(-3.0, 3.0)] * 16
            # Use DE with a small population centered around the best
            def _coord_de_obj(params):
                if time.time() > _DEADLINE:
                    return 1e10
                pts = params.reshape(8, 2)
                for i in range(8):
                    for j in range(i + 1, 8):
                        d = math.sqrt((pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2)
                        if d < 1e-10:
                            return 1e10
                ls = _log_score(pts)
                if not np.isfinite(ls):
                    return 1e10
                return -ls

            result = differential_evolution(
                _coord_de_obj,
                bounds=coord_bounds,
                seed=123,
                maxiter=100,
                popsize=10,
                tol=1e-8,
                mutation=(0.3, 1.0),
                recombination=0.7,
                init='sobol',
                polish=True,
                update_interval=1
            )
            if result.fun < 1e10:
                opt_pts = result.x.reshape(8, 2)
                ls_val = _log_score(opt_pts)
                if np.isfinite(ls_val) and ls_val > best_log_score:
                    best_log_score = ls_val
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
