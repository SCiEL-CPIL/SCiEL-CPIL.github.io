"""
Strategy: Global black-box search via differential evolution on Cartesian
coordinates, with smooth log-barriers for the diameter and separation
constraints, seeded from the antipodal-pair family (the known competitive
geometry for n=8) and a regular octagon.  This abandons the polynomial-
coefficient local optimization of the parent and instead uses a population-
based global method to escape shallow local minima, finishing with a
Nelder-Mead polish.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

_START_TIME = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill


def _elapsed():
    return time.time() - _START_TIME


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


def _neg_log_score(params):
    """
    Objective for differential evolution.
    params: 16 floats = [x0, y0, x1, y1, ..., x7, y7]
    We maximize log_score, so we minimize -log_score.
    Add smooth log-barriers:
      - diameter must be <= 2 (we want to keep it at 2 for the scaling, but
        the log_score already scales to diameter 2, so the constraint is
        really that points are well-separated and the diameter is not
        degenerate).  Actually, since log_score scales to diameter 2, the
        diameter constraint is automatically handled.  The only real
        constraint is that points are distinct (d > 0).
    We add a barrier for minimum pairwise distance to avoid coincident points.
    """
    if _elapsed() > _DEADLINE:
        return 1e10

    pts = params.reshape(8, 2)

    # Check for coincident points
    for i in range(8):
        for j in range(i + 1, 8):
            d = math.sqrt((pts[i][0] - pts[j][0])**2 + (pts[i][1] - pts[j][1])**2)
            if d < 1e-8:
                return 1e10

    ls = _log_score(pts)
    if not np.isfinite(ls):
        return 1e10

    # Add a soft barrier for very small pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    mask = ~np.eye(8, dtype=bool)
    valid_dists = dists[mask]
    min_d = np.min(valid_dists)
    if min_d < 0.05:
        barrier = 50.0 * (0.05 - min_d)**2
    else:
        barrier = 0.0

    return -ls + barrier


def _regular_octagon():
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _antipodal_init(rng):
    """
    Generate an antipodal-pair initialization: 4 pairs of antipodal points
    with random angles and radii, scaled to diameter 2.
    """
    base_angles = rng.uniform(0, np.pi / 2, 4)
    radii = rng.uniform(0.3, 1.5, 4)
    pts = []
    for i in range(4):
        a = base_angles[i]
        r = radii[i]
        pts.append([r * np.cos(a), r * np.sin(a)])
        pts.append([-r * np.cos(a), -r * np.sin(a)])
    pts = np.array(pts)
    d = np.sqrt(((pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2).sum(axis=2))
    max_d = np.max(d)
    if max_d > 0:
        pts = pts / max_d * 2.0
    return pts


def _two_squares_init(rng):
    """
    Generate a two-concentric-squares initialization.
    """
    r1 = rng.uniform(0.3, 1.0)
    r2 = rng.uniform(1.0, 1.8)
    rot1 = rng.uniform(0, np.pi / 4)
    rot2 = rng.uniform(0, np.pi / 4)
    pts = []
    for k in range(4):
        a = rot1 + 2 * np.pi * k / 4
        pts.append([r1 * np.cos(a), r1 * np.sin(a)])
    for k in range(4):
        a = rot2 + 2 * np.pi * k / 4
        pts.append([r2 * np.cos(a), r2 * np.sin(a)])
    pts = np.array(pts)
    d = np.sqrt(((pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2).sum(axis=2))
    max_d = np.max(d)
    if max_d > 0:
        pts = pts / max_d * 2.0
    return pts


def _nelder_mead_polish(pts, max_iter=3000):
    """Polish points with Nelder-Mead."""
    x0 = pts.flatten()
    try:
        res = minimize(_neg_log_score, x0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-10,
                                'fatol': 1e-10, 'adaptive': True})
        if res.fun < 1e10:
            opt_pts = res.x.reshape(8, 2)
            ls_val = _log_score(opt_pts)
            if np.isfinite(ls_val):
                return opt_pts, ls_val
    except Exception:
        pass
    return pts, _log_score(pts)


def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 140.0

    rng = np.random.default_rng(42)

    # --- Generate diverse initial points ---
    initial_pts_list = []

    # Regular octagon
    initial_pts_list.append(_regular_octagon())

    # Antipodal-pair initializations (the known competitive geometry)
    for _ in range(15):
        initial_pts_list.append(_antipodal_init(rng))

    # Two-concentric-squares initializations
    for _ in range(10):
        initial_pts_list.append(_two_squares_init(rng))

    # Random points
    for _ in range(10):
        radii = rng.uniform(0.1, 1.8, 8)
        angles_r = rng.uniform(0, 2 * np.pi, 8)
        pts = np.column_stack([radii * np.cos(angles_r), radii * np.sin(angles_r)])
        d = np.sqrt(((pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2).sum(axis=2))
        max_d = np.max(d)
        if max_d > 0:
            pts = pts / max_d * 2.0
        initial_pts_list.append(pts)

    # Score initial points
    scored_inits = []
    for pts in initial_pts_list:
        if _elapsed() > _DEADLINE - 60:
            break
        ls = _log_score(pts)
        if np.isfinite(ls):
            scored_inits.append((ls, pts))

    scored_inits.sort(key=lambda x: x[0], reverse=True)

    best_points = scored_inits[0][1] if scored_inits else _regular_octagon()
    best_log_score = scored_inits[0][0] if scored_inits else _log_score(best_points)

    # --- Phase 1: Differential Evolution (global search) ---
    # Bounds: coordinates in [-3, 3] (generous; the scaling handles the rest)
    bounds = [(-3.0, 3.0)] * 16

    # Run DE on the top initializations
    top_inits = scored_inits[:8]

    for ls0, pts0 in top_inits:
        if _elapsed() > _DEADLINE - 40:
            break
        try:
            # Differential evolution
            de_res = differential_evolution(
                _neg_log_score,
                bounds=bounds,
                x0=pts0.flatten(),
                seed=42,
                maxiter=80,
                tol=1e-8,
                popsize=15,
                polish=True,
                init='sobol',
                updating='immediate'
            )
            if de_res.fun < 1e10:
                opt_pts = de_res.x.reshape(8, 2)
                ls_val = _log_score(opt_pts)
                if np.isfinite(ls_val) and ls_val > best_log_score:
                    best_log_score = ls_val
                    best_points = opt_pts
        except Exception:
            pass

    # --- Phase 2: Nelder-Mead polish on the best and top candidates ---
    # Polish the best
    if _elapsed() < _DEADLINE - 10:
        polished_pts, polished_ls = _nelder_mead_polish(best_points, max_iter=5000)
        if polished_ls > best_log_score:
            best_log_score = polished_ls
            best_points = polished_pts

    # Polish top-5 initializations
    for ls0, pts0 in scored_inits[:5]:
        if _elapsed() > _DEADLINE - 5:
            break
        polished_pts, polished_ls = _nelder_mead_polish(pts0, max_iter=3000)
        if polished_ls > best_log_score:
            best_log_score = polished_ls
            best_points = polished_pts

    # Final polish on the best
    if _elapsed() < _DEADLINE - 2:
        polished_pts, polished_ls = _nelder_mead_polish(best_points, max_iter=5000)
        if polished_ls > best_log_score:
            best_log_score = polished_ls
            best_points = polished_pts

    # Ensure coordinates are within bounds
    final_result = []
    for p in best_points:
        x = max(-1e6, min(1e6, float(p[0])))
        y = max(-1e6, min(1e6, float(p[1])))
        final_result.append([x, y])

    return final_result
