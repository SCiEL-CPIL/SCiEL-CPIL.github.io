"""
Strategy: Direct coordinate optimization via a smooth log-sum-exp diameter
penalty, with differential evolution for global exploration and L-BFGS-B
for local refinement.

This is structurally different from the polynomial-discriminant approach:
- No root-finding, no coefficient space, no Sylvester matrices.
- The objective is computed directly from 16 coordinates.
- The non-differentiable max() in the diameter is replaced by a smooth
  log-sum-exp approximation, making the entire problem smooth in 16D.
- Differential evolution provides a different global search mechanism
  (population-based, mutation/crossover) vs. dual annealing (simulated-annealing
  style with restarts).
"""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

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


def _neg_smooth_log_score(params, beta):
    """
    Negative smooth log score. Uses log-sum-exp to approximate the max distance
    smoothly, avoiding the kink in the non-smooth max().
    
    params: 16-element array (8 points, 2 coords each)
    beta: smoothness parameter (higher = closer to true max)
    """
    if time.time() > _DEADLINE:
        return 1e10
    pts = params.reshape(8, 2)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    mask = ~np.eye(8, dtype=bool)
    valid_dists = dists[mask]
    if np.any(valid_dists <= 0):
        return 1e10
    max_d = np.max(valid_dists)
    if max_d <= 0:
        return 1e10
    # Smooth approximation of max via log-sum-exp
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


def _neg_log_score(params):
    """Negative true log score for final evaluation."""
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


def _two_concentric_squares(r1, r2, rot=0.0):
    """Two concentric squares with radii r1, r2 and rotation rot for the outer."""
    pts = []
    for k in range(4):
        a = 2 * np.pi * k / 4
        pts.append([r1 * np.cos(a), r1 * np.sin(a)])
    for k in range(4):
        a = rot + 2 * np.pi * k / 4
        pts.append([r2 * np.cos(a), r2 * np.sin(a)])
    return np.array(pts)


def _antipodal_pairs(angles, radii):
    """4 antipodal pairs."""
    pts = []
    for i in range(4):
        a = angles[i]
        r = radii[i]
        pts.append([r * np.cos(a), r * np.sin(a)])
        pts.append([-r * np.cos(a), -r * np.sin(a)])
    return np.array(pts)


def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 135.0

    best_points = _regular_octagon()
    best_log_score = _log_score(best_points)

    rng = np.random.default_rng(42)

    # === Stage 1: Generate diverse initial candidates ===
    init_candidates = []

    # Regular octagon
    init_candidates.append(_regular_octagon().flatten())

    # Two concentric squares with various parameters
    for r1 in [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]:
        for r2 in [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]:
            if abs(r1 - r2) < 0.2:
                continue
            for rot in [0.0, np.pi / 8, np.pi / 4, np.pi / 6, np.pi / 3]:
                pts = _two_concentric_squares(r1, r2, rot)
                init_candidates.append(pts.flatten())

    # Antipodal pairs
    for _ in range(15):
        angles = rng.uniform(0, np.pi / 2, 4)
        radii = rng.uniform(0.5, 1.5, 4)
        pts = _antipodal_pairs(angles, radii)
        init_candidates.append(pts.flatten())

    # Random points
    for _ in range(20):
        radii = rng.uniform(0.3, 1.8, 8)
        angles = rng.uniform(0, 2 * np.pi, 8)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        init_candidates.append(pts.flatten())

    # Score initial candidates
    scored_inits = []
    for cand in init_candidates:
        if time.time() > _DEADLINE - 80:
            break
        try:
            ls = _log_score(cand.reshape(8, 2))
            if np.isfinite(ls):
                scored_inits.append((ls, cand))
        except Exception:
            pass

    scored_inits.sort(key=lambda x: x[0], reverse=True)

    # Update best
    for ls, cand in scored_inits:
        if ls > best_log_score:
            best_log_score = ls
            best_points = cand.reshape(8, 2)

    # === Stage 2: Differential evolution for global search ===
    # Use a moderate population size and a smooth objective with beta=100
    beta_de = 100.0
    bounds_de = [(-3.0, 3.0)] * 16

    if time.time() < _DEADLINE - 60:
        try:
            de_result = differential_evolution(
                lambda x: _neg_smooth_log_score(x, beta_de),
                bounds=bounds_de,
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-10,
                mutation=(0.3, 1.2),
                recombination=0.8,
                polish=False,
                init='sobol'
            )
            if de_result.fun < 1e10:
                de_pts = de_result.x.reshape(8, 2)
                ls_val = _log_score(de_pts)
                if np.isfinite(ls_val) and ls_val > best_log_score:
                    best_log_score = ls_val
                    best_points = de_pts
        except Exception:
            pass

    # === Stage 3: Multi-start local refinement with increasing beta ===
    # Collect a pool of candidates from DE and top initial candidates
    candidate_pool = [best_points.flatten()]
    for ls, cand in scored_inits[:10]:
        candidate_pool.append(cand)

    # Add some perturbed versions of the best
    for _ in range(10):
        perturbed = best_points.flatten() + rng.normal(0, 0.1, 16)
        candidate_pool.append(perturbed)

    # Refine each candidate with L-BFGS-B using smooth objective
    for beta in [50.0, 200.0, 500.0, 1000.0]:
        if time.time() > _DEADLINE - 15:
            break
        for cand in candidate_pool[:15]:
            if time.time() > _DEADLINE - 10:
                break
            try:
                res = minimize(
                    lambda x, b=beta: _neg_smooth_log_score(x, b),
                    cand,
                    method='L-BFGS-B',
                    bounds=bounds_de,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.fun < 1e10:
                    refined = res.x.reshape(8, 2)
                    ls_val = _log_score(refined)
                    if np.isfinite(ls_val) and ls_val > best_log_score:
                        best_log_score = ls_val
                        best_points = refined
                        candidate_pool[0] = refined.flatten()
            except Exception:
                pass

    # === Stage 4: Final Nelder-Mead polish on the best ===
    if time.time() < _DEADLINE - 3:
        try:
            res = minimize(
                _neg_log_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
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
