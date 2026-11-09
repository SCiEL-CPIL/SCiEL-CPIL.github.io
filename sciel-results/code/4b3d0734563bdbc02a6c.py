"""Heilbronn convex regions: 13 points maximizing min triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
DEADLINE = 145.0  # seconds wall-clock limit; keep safety margin

# Precompute all triple indices
_ALL_TRIPLES = np.array(list(combinations(range(N), 3)))


def _all_triangle_areas(points):
    """Compute all triangle areas as a vector."""
    p1 = points[_ALL_TRIPLES[:, 0]]
    p2 = points[_ALL_TRIPLES[:, 1]]
    p3 = points[_ALL_TRIPLES[:, 2]]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    return 0.5 * np.abs(cross)


def _min_triangle_area(points):
    """Compute the minimum triangle area among all triples of points."""
    if len(points) < 3:
        return 0.0
    return float(np.min(_all_triangle_areas(points)))


def _hull_area(points):
    """Compute the convex hull area of the point set."""
    try:
        hull = ConvexHull(points)
        return float(hull.volume)  # In 2D, volume is area
    except Exception:
        return 1.0


def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    return _min_triangle_area(points) / hull_area


def _negative_score(points):
    return -_score(points)


def _soft_min_triangle_area(points, beta=200.0):
    """
    Smooth surrogate for min triangle area using log-sum-exp (soft-min).
    soft_min(a) = -(1/beta) * log(sum(exp(-beta * a_i)))
    As beta -> inf, this approaches the true min.
    """
    areas = _all_triangle_areas(points)
    areas = np.clip(areas, 1e-15, None)
    neg_beta_a = -beta * areas
    max_val = np.max(neg_beta_a)
    log_sum = max_val + np.log(np.sum(np.exp(neg_beta_a - max_val)))
    soft_min = -log_sum / beta
    return float(soft_min)


def _soft_objective(x, beta=200.0):
    """
    Smooth objective: -soft_min_triangle_area / hull_area.
    Returns a smooth scalar to be minimized.
    """
    pts = x.reshape(-1, 2)
    hull_area = _hull_area(pts)
    if hull_area < 1e-8:
        return 1e10
    sm = _soft_min_triangle_area(pts, beta=beta)
    if sm < 1e-15:
        return 1e10
    return -sm / hull_area


def _exact_objective(x):
    """Exact nonsmooth objective for final polish."""
    pts = x.reshape(-1, 2)
    hull_area = _hull_area(pts)
    if hull_area < 1e-8:
        return 1e10
    min_area = _min_triangle_area(pts)
    return -min_area / hull_area


def _regular_polygon(n, perturb=0.0, rng_seed=42):
    """Create a regular n-gon, optionally perturbed."""
    rng = np.random.RandomState(rng_seed)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    if perturb > 0:
        pts += rng.randn(n, 2) * perturb
    return pts


def construct():
    start_time = time.time()
    best_points = None
    best_score = -np.inf

    def _time_remaining():
        return DEADLINE - (time.time() - start_time)

    def _update_best(pts):
        nonlocal best_points, best_score
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
        return s

    # === Phase 0: Build diverse initial configurations ===
    configs = []

    # Regular 13-gon
    configs.append(_regular_polygon(13))

    # Perturbed 13-gons
    for perturb in [0.01, 0.02, 0.03, 0.05, 0.08, 0.12]:
        for seed in range(10):
            configs.append(_regular_polygon(13, perturb=perturb, rng_seed=seed))

    # 12-gon + center
    for seed in range(10):
        outer = _regular_polygon(12, perturb=0.02, rng_seed=seed)
        center = np.random.RandomState(seed).randn(2) * 0.05
        configs.append(np.vstack([outer, np.array([center])]))

    # 11-gon + 2 interior
    for seed in range(8):
        outer = _regular_polygon(11, perturb=0.02, rng_seed=seed)
        rng = np.random.RandomState(seed)
        interior = rng.randn(2, 2) * 0.05
        configs.append(np.vstack([outer, interior]))

    # 10-gon + 3 interior
    for seed in range(8):
        outer = _regular_polygon(10, perturb=0.02, rng_seed=seed)
        rng = np.random.RandomState(seed)
        interior = rng.randn(3, 2) * 0.06
        configs.append(np.vstack([outer, interior]))

    # Random configurations
    for seed in range(30):
        rng = np.random.RandomState(1000 + seed)
        pts = rng.randn(13, 2)
        configs.append(pts)

    # Evaluate all configs
    for cfg in configs:
        if _time_remaining() < 5.0:
            break
        _update_best(cfg)

    # Sort by score
    scored_configs = []
    for cfg in configs:
        s = _score(cfg)
        scored_configs.append((s, cfg))
    scored_configs.sort(key=lambda x: -x[0])

    # === Phase 1: L-BFGS-B with progressive soft-min beta ===
    betas = [20.0, 40.0, 80.0, 150.0, 300.0, 600.0, 1200.0]

    for beta in betas:
        if _time_remaining() < 15.0:
            break
        n_starts = min(8, len(scored_configs))
        for s, cfg in scored_configs[:n_starts]:
            if _time_remaining() < 5.0:
                break
            x0 = cfg.flatten()
            try:
                result = minimize(
                    _soft_objective, x0, args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                opt_pts = result.x.reshape(-1, 2)
                _update_best(opt_pts)
            except Exception:
                pass

    # === Phase 2: Nelder-Mead polish from best ===
    if _time_remaining() > 15.0 and best_points is not None:
        x0 = best_points.flatten()
        try:
            result = minimize(
                _exact_objective, x0, method='Nelder-Mead',
                options={'maxiter': 8000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            opt_pts = result.x.reshape(-1, 2)
            _update_best(opt_pts)
        except Exception:
            pass

    # === Phase 3: Powell method (coordinate-based, good for nonsmooth) ===
    if _time_remaining() > 12.0 and best_points is not None:
        x0 = best_points.flatten()
        try:
            result = minimize(
                _exact_objective, x0, method='Powell',
                options={'maxiter': 4000, 'ftol': 1e-13, 'xtol': 1e-11}
            )
            opt_pts = result.x.reshape(-1, 2)
            _update_best(opt_pts)
        except Exception:
            pass

    # === Phase 4: Nelder-Mead from other good starts ===
    if _time_remaining() > 10.0:
        for s, cfg in scored_configs[:5]:
            if _time_remaining() < 5.0:
                break
            x0 = cfg.flatten()
            try:
                result = minimize(
                    _exact_objective, x0, method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                opt_pts = result.x.reshape(-1, 2)
                _update_best(opt_pts)
            except Exception:
                pass

    # === Phase 5: High-precision L-BFGS-B with sharp soft-min ===
    if _time_remaining() > 8.0 and best_points is not None:
        x0 = best_points.flatten()
        for beta in [2000.0, 5000.0]:
            if _time_remaining() < 4.0:
                break
            try:
                result = minimize(
                    _soft_objective, x0, args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                opt_pts = result.x.reshape(-1, 2)
                _update_best(opt_pts)
                x0 = opt_pts.flatten()
            except Exception:
                pass

    # === Phase 6: Final Nelder-Mead with tight tolerances ===
    if _time_remaining() > 5.0 and best_points is not None:
        x0 = best_points.flatten()
        try:
            result = minimize(
                _exact_objective, x0, method='Nelder-Mead',
                options={'maxiter': 6000, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True}
            )
            opt_pts = result.x.reshape(-1, 2)
            _update_best(opt_pts)
        except Exception:
            pass

    # === Phase 7: One more Powell pass ===
    if _time_remaining() > 3.0 and best_points is not None:
        x0 = best_points.flatten()
        try:
            result = minimize(
                _exact_objective, x0, method='Powell',
                options={'maxiter': 3000, 'ftol': 1e-14, 'xtol': 1e-12}
            )
            opt_pts = result.x.reshape(-1, 2)
            _update_best(opt_pts)
        except Exception:
            pass

    # === Phase 8: Final Nelder-Mead ===
    if _time_remaining() > 2.0 and best_points is not None:
        x0 = best_points.flatten()
        try:
            result = minimize(
                _exact_objective, x0, method='Nelder-Mead',
                options={'maxiter': 4000, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True}
            )
            opt_pts = result.x.reshape(-1, 2)
            _update_best(opt_pts)
        except Exception:
            pass

    if best_points is None:
        # Fallback: regular 13-gon
        best_points = _regular_polygon(13)

    return best_points
