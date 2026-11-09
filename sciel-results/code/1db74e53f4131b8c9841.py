"""Heilbronn convex regions: 13 points maximizing min triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
DEADLINE = 115.0  # seconds wall-clock limit; keep safety margin

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
    # Clip to avoid overflow
    areas = np.clip(areas, 1e-15, None)
    neg_beta_a = -beta * areas
    # Shift for numerical stability
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


def construct():
    start_time = time.time()
    best_points = None
    best_score = -np.inf

    def _time_remaining():
        return DEADLINE - (time.time() - start_time)

    def _regular_12_plus_center(perturb=0.0, center_offset=0.0, rng_seed=42):
        rng = np.random.RandomState(rng_seed)
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        if perturb > 0:
            pts += rng.randn(12, 2) * perturb
        center = np.array([center_offset, center_offset])
        all_pts = np.vstack([pts, center])
        return all_pts

    # Build diverse initial configurations
    configs = []

    # Regular 12-gon + center
    configs.append(_regular_12_plus_center(perturb=0.0, center_offset=0.0, rng_seed=42))

    # Offset centers
    for dx in [-0.1, 0.1, -0.05, 0.05, 0.0]:
        for dy in [-0.1, 0.1, -0.05, 0.05, 0.0]:
            if dx == 0 and dy == 0:
                continue
            configs.append(_regular_12_plus_center(perturb=0.0, center_offset=dx, rng_seed=42 + int(abs(dx*100) + abs(dy*100))))

    # Perturbed 12-gons
    for perturb in [0.01, 0.02, 0.03, 0.05, 0.08]:
        for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            configs.append(_regular_12_plus_center(perturb=perturb, center_offset=0.0, rng_seed=seed))

    # Random near 12-gon with varying radii
    for seed in range(30):
        rng2 = np.random.RandomState(seed)
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False) + rng2.randn(12) * 0.02
        radii = 1.0 + rng2.randn(12) * 0.06
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        center = rng2.randn(2) * 0.06
        all_pts = np.vstack([pts, center])
        configs.append(all_pts)

    # Evaluate all configs with exact score
    for cfg in configs:
        if _time_remaining() < 5.0:
            break
        s = _score(cfg)
        if s > best_score:
            best_score = s
            best_points = cfg.copy()

    # Phase 1: L-BFGS-B with soft-min objective from multiple starts
    # Use progressively larger beta for sharper approximation
    betas = [50.0, 100.0, 200.0, 500.0]
    
    for beta in betas:
        if _time_remaining() < 20.0:
            break
        # Optimize from top-k starting points
        scored_configs = []
        for cfg in configs:
            s = _score(cfg)
            scored_configs.append((s, cfg))
        scored_configs.sort(key=lambda x: -x[0])
        
        for s, cfg in scored_configs[:8]:
            if _time_remaining() < 8.0:
                break
            x0 = cfg.flatten()
            try:
                result = minimize(
                    _soft_objective, x0, args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                opt_pts = result.x.reshape(-1, 2)
                opt_score = _score(opt_pts)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_pts.copy()
            except Exception:
                pass

    # Phase 2: Nelder-Mead polish with exact objective from best result
    if _time_remaining() > 15.0:
        x0 = best_points.flatten()
        try:
            result = minimize(
                _exact_objective, x0, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            opt_pts = result.x.reshape(-1, 2)
            opt_score = _score(opt_pts)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
        except Exception:
            pass

    # Phase 3: Additional Nelder-Mead from other good starts
    if _time_remaining() > 10.0:
        scored_configs = []
        for cfg in configs[:15]:
            s = _score(cfg)
            scored_configs.append((s, cfg))
        scored_configs.sort(key=lambda x: -x[0])
        
        for s, cfg in scored_configs[:5]:
            if _time_remaining() < 5.0:
                break
            x0 = cfg.flatten()
            try:
                result = minimize(
                    _exact_objective, x0, method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                opt_pts = result.x.reshape(-1, 2)
                opt_score = _score(opt_pts)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_pts.copy()
            except Exception:
                pass

    # Phase 4: Final high-precision L-BFGS-B polish with very sharp soft-min
    if _time_remaining() > 5.0:
        x0 = best_points.flatten()
        try:
            result = minimize(
                _soft_objective, x0, args=(1000.0,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            opt_pts = result.x.reshape(-1, 2)
            opt_score = _score(opt_pts)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
        except Exception:
            pass

    # Phase 5: Final exact Nelder-Mead with tight tolerances
    if _time_remaining() > 3.0:
        x0 = best_points.flatten()
        try:
            result = minimize(
                _exact_objective, x0, method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            opt_pts = result.x.reshape(-1, 2)
            opt_score = _score(opt_pts)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
        except Exception:
            pass

    return best_points
