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

    def _make_regular_ngon(n, center=None, radius=1.0, phase=0.0):
        angles = np.linspace(phase, phase + 2 * np.pi, n, endpoint=False)
        if center is None:
            center = np.array([0.0, 0.0])
        return np.column_stack((center[0] + radius * np.cos(angles),
                                center[1] + radius * np.sin(angles)))

    # Build diverse initial configurations
    configs = []

    # 1. Regular 13-gon
    configs.append(_make_regular_ngon(13, radius=1.0))

    # 2. Regular 12-gon + center
    ring12 = _make_regular_ngon(12, radius=1.0)
    configs.append(np.vstack([ring12, np.array([[0.0, 0.0]])]))

    # 3. Offset centers for 12-gon + center
    for dx in [-0.1, 0.1, -0.05, 0.05]:
        for dy in [-0.1, 0.1, -0.05, 0.05]:
            if dx == 0 and dy == 0:
                continue
            ring12b = _make_regular_ngon(12, radius=1.0)
            configs.append(np.vstack([ring12b, np.array([[dx, dy]])]))

    # 4. Regular 11-gon + 2 interior
    ring11 = _make_regular_ngon(11, radius=1.0)
    interior2 = np.array([[0.3, 0.0], [-0.3, 0.0]])
    configs.append(np.vstack([ring11, interior2]))

    # 5. 7 outer + 6 inner concentric
    outer7 = _make_regular_ngon(7, radius=1.0)
    inner6 = _make_regular_ngon(6, radius=0.55, phase=np.pi / 6)
    configs.append(np.vstack([outer7, inner6]))

    # 6. 9 outer + 4 inner (square)
    outer9 = _make_regular_ngon(9, radius=1.0)
    inner4 = np.array([[0.45, 0.45], [-0.45, 0.45], [-0.45, -0.45], [0.45, -0.45]])
    configs.append(np.vstack([outer9, inner4]))

    # 7. 10 outer + 3 inner (triangle)
    outer10 = _make_regular_ngon(10, radius=1.0)
    inner3 = _make_regular_ngon(3, radius=0.45)
    configs.append(np.vstack([outer10, inner3]))

    # 8. 8 outer + 5 inner
    outer8 = _make_regular_ngon(8, radius=1.0)
    inner5 = _make_regular_ngon(5, radius=0.5, phase=np.pi / 5)
    configs.append(np.vstack([outer8, inner5]))

    # 9. Perturbed regular 13-gon
    base = _make_regular_ngon(13, radius=1.0)
    rng = np.random.RandomState(42)
    perturbed = base + rng.normal(0, 0.05, base.shape)
    configs.append(perturbed)

    # 10. 12-gon with center and one vertex slightly inward
    ring12c = _make_regular_ngon(12, radius=1.0)
    ring12c[0] = ring12c[0] * 0.9
    configs.append(np.vstack([ring12c, np.array([[0.0, 0.0]])]))

    # 11. Grid-like structure
    grid = np.array([
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0, 1], [1, 1], [2, 1], [3, 1],
        [0, 2], [1, 2], [2, 2], [3, 2],
        [0, 3], [1, 3], [2, 3], [3, 3]
    ])
    grid_pts = grid[:13] - np.mean(grid[:13], axis=0)
    scale = 1.0 / np.max(np.linalg.norm(grid_pts, axis=1))
    configs.append(grid_pts * scale)

    # 12. Random cluster
    random_pts = rng.normal(0, 0.5, (N, 2))
    random_pts -= np.mean(random_pts, axis=0)
    scale = 1.0 / np.max(np.linalg.norm(random_pts, axis=1))
    configs.append(random_pts * scale)

    # 13. Elliptical distribution
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    ellipse = np.column_stack((1.2 * np.cos(angles), 0.8 * np.sin(angles)))
    configs.append(ellipse)

    # 14. Two concentric 7-gons with offset
    outer7b = _make_regular_ngon(7, radius=1.0)
    inner7 = _make_regular_ngon(7, radius=0.5, phase=np.pi / 7)
    configs.append(np.vstack([outer7b, inner7[:6]]))

    # 15. 13 points on a circle with radial jitter
    base13 = _make_regular_ngon(13, radius=1.0)
    radii = 1.0 + rng.normal(0, 0.08, N)
    angles13 = np.linspace(0, 2 * np.pi, N, endpoint=False)
    jittered = np.column_stack((radii * np.cos(angles13), radii * np.sin(angles13)))
    configs.append(jittered)

    # 16. 13 points with alternating radii
    radii_alt = np.array([1.0, 0.85] * 6 + [1.0])
    angles13b = np.linspace(0, 2 * np.pi, N, endpoint=False)
    alt_pts = np.column_stack((radii_alt * np.cos(angles13b), radii_alt * np.sin(angles13b)))
    configs.append(alt_pts)

    # 17. 13 points on a superellipse-like shape
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    n_sup = 4
    sup_x = np.sign(np.cos(t)) * np.abs(np.cos(t)) ** (2.0 / n_sup)
    sup_y = np.sign(np.sin(t)) * np.abs(np.sin(t)) ** (2.0 / n_sup)
    sup_pts = np.column_stack((sup_x, sup_y))
    configs.append(sup_pts)

    # 18. 13 points: 6 outer hexagon + 7 inner heptagon
    hex_outer = _make_regular_ngon(6, radius=1.0)
    hep_inner = _make_regular_ngon(7, radius=0.55, phase=np.pi / 7)
    configs.append(np.vstack([hex_outer, hep_inner]))

    # 19. 13 points: 5 outer pentagon + 8 inner
    pent_outer = _make_regular_ngon(5, radius=1.0)
    inner8 = _make_regular_ngon(8, radius=0.6, phase=np.pi / 8)
    configs.append(np.vstack([pent_outer, inner8]))

    # 20. 13 points: 4 outer square + 9 inner
    sq_outer = _make_regular_ngon(4, radius=1.0, phase=np.pi / 4)
    inner9 = _make_regular_ngon(9, radius=0.65)
    configs.append(np.vstack([sq_outer, inner9]))

    # 21. Hexagonal lattice pattern
    hex_pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i + 0.5 * j
            y = j * np.sqrt(3) / 2
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts[:13])
    hex_pts -= np.mean(hex_pts, axis=0)
    scale = 1.0 / np.max(np.linalg.norm(hex_pts, axis=1))
    configs.append(hex_pts * scale)

    # 22. More perturbed 12-gons
    for perturb in [0.01, 0.02, 0.03, 0.05]:
        for seed in [1, 2, 3, 4]:
            rng2 = np.random.RandomState(seed)
            angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
            pts = np.column_stack([np.cos(angles), np.sin(angles)])
            if perturb > 0:
                pts += rng2.randn(12, 2) * perturb
            center = rng2.randn(2) * 0.05
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

    # Sort configs by score for prioritization
    scored_configs = []
    for cfg in configs:
        s = _score(cfg)
        scored_configs.append((s, cfg))
    scored_configs.sort(key=lambda x: -x[0])

    # Phase 1: L-BFGS-B with soft-min objective from multiple starts
    # Use progressively larger beta for sharper approximation
    betas = [50.0, 100.0, 200.0, 500.0, 800.0]
    
    for beta in betas:
        if _time_remaining() < 15.0:
            break
        for s, cfg in scored_configs[:6]:
            if _time_remaining() < 6.0:
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
        for s, cfg in scored_configs[:4]:
            if _time_remaining() < 4.0:
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

    # Phase 5: Local refinement with small perturbations around the best point
    if best_points is not None and _time_remaining() > 10.0:
        x_best = best_points.flatten()
        for pert_scale in [0.005, 0.01, 0.02]:
            if _time_remaining() < 5.0:
                break
            for _ in range(3):
                if _time_remaining() < 3.0:
                    break
                x_pert = x_best + rng.normal(0, pert_scale, len(x_best))
                try:
                    res_p = minimize(_soft_objective, x_pert, args=(800,),
                                     method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12})
                    opt_pts = res_p.x.reshape(-1, 2)
                    opt_score = _score(opt_pts)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_points = opt_pts.copy()
                        x_best = res_p.x
                except Exception:
                    pass

    # Phase 6: Coordinate-wise exact polish
    if best_points is not None and _time_remaining() > 8.0:
        pts = best_points.copy()
        improved = True
        max_rounds = 2
        for _round in range(max_rounds):
            if not improved or _time_remaining() < 4.0:
                break
            improved = False
            for i in range(N):
                if _time_remaining() < 2.0:
                    break
                for d in range(2):
                    best_coord = pts[i, d]
                    best_val = _score(pts)
                    for step in [0.001, -0.001, 0.0005, -0.0005, 0.002, -0.002]:
                        pts[i, d] = best_coord + step
                        s = _score(pts)
                        if s > best_val:
                            best_val = s
                            best_coord = pts[i, d]
                    pts[i, d] = best_coord
                    if best_val > best_score:
                        best_score = best_val
                        best_points = pts.copy()
                        improved = True

    # Final fallback: regular polygon
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = _score(best_points)

    return best_points
