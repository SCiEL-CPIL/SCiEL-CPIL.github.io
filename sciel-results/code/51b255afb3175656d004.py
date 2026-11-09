import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def compute_min_triangle_area(points):
    """Compute minimum triangle area among all triples."""
    n = len(points)
    if n < 3:
        return 0.0
    idx = np.arange(n)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i < j) & (j < k)
    i = i[mask]
    j = j[mask]
    k = k[mask]
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def get_score(points):
    """Compute score: min triangle area / convex hull area."""
    N = len(points)
    if N < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except:
        return 0.0
    min_area = compute_min_triangle_area(points)
    if min_area <= 0:
        return 0.0
    return min_area / hull_area

def soft_min(areas, tau):
    """Soft minimum with log-sum-exp stability and clipping."""
    min_val = np.min(areas)
    shifted = areas - min_val
    shifted = np.maximum(shifted, -50.0 / tau)
    log_sum = np.log(np.sum(np.exp(-tau * shifted)))
    return min_val - (1.0 / tau) * log_sum

def compute_all_areas(points):
    """Compute all triangle areas for n points (vectorized)."""
    n = len(points)
    idx = np.arange(n)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i < j) & (j < k)
    i = i[mask]
    j = j[mask]
    k = k[mask]
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas

def construct():
    start_time = time.time()
    deadline = start_time + 142.0
    N = 13
    best_score = 0.0
    best_points = None
    rng = np.random.RandomState(42)

    def time_left():
        return deadline - time.time()

    def params_to_points(params):
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(N - 2, 2)
        return pts

    def points_to_params(pts):
        p0 = pts[0]
        p1 = pts[1]
        d = p1 - p0
        dist = np.linalg.norm(d)
        if dist < 1e-10:
            return np.zeros(2 * (N - 2))
        pts_t = pts - p0
        angle = np.arctan2(d[1], d[0])
        c, s = np.cos(-angle), np.sin(-angle)
        R = np.array([[c, -s], [s, c]])
        pts_r = pts_t @ R.T
        pts_s = pts_r / dist
        return pts_s[2:].flatten()

    def smooth_objective(params, tau):
        try:
            pts = params_to_points(params)
            areas = compute_all_areas(pts)
            sm = soft_min(areas, tau)
            try:
                hull = ConvexHull(pts)
                hull_area = hull.volume
                if hull_area < 1e-10:
                    return 1e6
            except:
                return 1e6
            score = sm / hull_area
            return -score
        except:
            return 1e6

    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        try:
            s = get_score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except:
            pass

    def make_regular_ngon(n, center=None, radius=1.0, phase=0.0):
        angles = np.linspace(phase, phase + 2 * np.pi, n, endpoint=False)
        if center is None:
            center = np.array([0.0, 0.0])
        return np.column_stack((center[0] + radius * np.cos(angles),
                                center[1] + radius * np.sin(angles)))

    # Generate diverse initial configurations
    initial_configs = []

    # 1. Regular 13-gon
    initial_configs.append(make_regular_ngon(13, radius=1.0))

    # 2. Regular 12-gon + center
    ring12 = make_regular_ngon(12, radius=1.0)
    initial_configs.append(np.vstack([ring12, np.array([[0.0, 0.0]])]))

    # 3. Regular 11-gon + 2 interior
    ring11 = make_regular_ngon(11, radius=1.0)
    interior2 = np.array([[0.3, 0.0], [-0.3, 0.0]])
    initial_configs.append(np.vstack([ring11, interior2]))

    # 4. 7 outer + 6 inner concentric
    outer7 = make_regular_ngon(7, radius=1.0)
    inner6 = make_regular_ngon(6, radius=0.55, phase=np.pi / 6)
    initial_configs.append(np.vstack([outer7, inner6]))

    # 5. 9 outer + 4 inner (square)
    outer9 = make_regular_ngon(9, radius=1.0)
    inner4 = np.array([[0.45, 0.45], [-0.45, 0.45], [-0.45, -0.45], [0.45, -0.45]])
    initial_configs.append(np.vstack([outer9, inner4]))

    # 6. 10 outer + 3 inner (triangle)
    outer10 = make_regular_ngon(10, radius=1.0)
    inner3 = make_regular_ngon(3, radius=0.45)
    initial_configs.append(np.vstack([outer10, inner3]))

    # 7. 8 outer + 5 inner
    outer8 = make_regular_ngon(8, radius=1.0)
    inner5 = make_regular_ngon(5, radius=0.5, phase=np.pi / 5)
    initial_configs.append(np.vstack([outer8, inner5]))

    # 8. Perturbed regular polygon
    base = make_regular_ngon(13, radius=1.0)
    perturbed = base + rng.normal(0, 0.05, base.shape)
    initial_configs.append(perturbed)

    # 9. 12-gon with center pushed off-center
    ring12b = make_regular_ngon(12, radius=1.0)
    initial_configs.append(np.vstack([ring12b, np.array([[0.15, 0.0]])]))

    # 10. 12-gon with center and one vertex slightly inward
    ring12c = make_regular_ngon(12, radius=1.0)
    ring12c[0] = ring12c[0] * 0.9
    initial_configs.append(np.vstack([ring12c, np.array([[0.0, 0.0]])]))

    # 11. Grid-like structure
    grid = np.array([
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0, 1], [1, 1], [2, 1], [3, 1],
        [0, 2], [1, 2], [2, 2], [3, 2],
        [0, 3], [1, 3], [2, 3], [3, 3]
    ])
    grid_pts = grid[:13] - np.mean(grid[:13], axis=0)
    scale = 1.0 / np.max(np.linalg.norm(grid_pts, axis=1))
    initial_configs.append(grid_pts * scale)

    # 12. Random cluster
    random_pts = rng.normal(0, 0.5, (N, 2))
    random_pts -= np.mean(random_pts, axis=0)
    scale = 1.0 / np.max(np.linalg.norm(random_pts, axis=1))
    initial_configs.append(random_pts * scale)

    # 13. Elliptical distribution
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    ellipse = np.column_stack((1.2 * np.cos(angles), 0.8 * np.sin(angles)))
    initial_configs.append(ellipse)

    # 14. Two concentric 7-gons with offset
    outer7b = make_regular_ngon(7, radius=1.0)
    inner7 = make_regular_ngon(7, radius=0.5, phase=np.pi / 7)
    initial_configs.append(np.vstack([outer7b, inner7[:6]]))

    # 15. 13 points on a circle with radial jitter
    base13 = make_regular_ngon(13, radius=1.0)
    radii = 1.0 + rng.normal(0, 0.08, N)
    angles13 = np.linspace(0, 2 * np.pi, N, endpoint=False)
    jittered = np.column_stack((radii * np.cos(angles13), radii * np.sin(angles13)))
    initial_configs.append(jittered)

    # 16. 13 points with alternating radii
    radii_alt = np.array([1.0, 0.85] * 6 + [1.0])
    angles13b = np.linspace(0, 2 * np.pi, N, endpoint=False)
    alt_pts = np.column_stack((radii_alt * np.cos(angles13b), radii_alt * np.sin(angles13b)))
    initial_configs.append(alt_pts)

    # 17. 13 points on a superellipse-like shape
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    n_sup = 4
    sup_x = np.sign(np.cos(t)) * np.abs(np.cos(t)) ** (2.0 / n_sup)
    sup_y = np.sign(np.sin(t)) * np.abs(np.sin(t)) ** (2.0 / n_sup)
    sup_pts = np.column_stack((sup_x, sup_y))
    initial_configs.append(sup_pts)

    # 18. 13 points: 6 outer hexagon + 7 inner heptagon
    hex_outer = make_regular_ngon(6, radius=1.0)
    hep_inner = make_regular_ngon(7, radius=0.55, phase=np.pi / 7)
    initial_configs.append(np.vstack([hex_outer, hep_inner]))

    # 19. 13 points: 5 outer pentagon + 8 inner
    pent_outer = make_regular_ngon(5, radius=1.0)
    inner8 = make_regular_ngon(8, radius=0.6, phase=np.pi / 8)
    initial_configs.append(np.vstack([pent_outer, inner8]))

    # 20. 13 points: 4 outer square + 9 inner
    sq_outer = make_regular_ngon(4, radius=1.0, phase=np.pi / 4)
    inner9 = make_regular_ngon(9, radius=0.65)
    initial_configs.append(np.vstack([sq_outer, inner9]))

    # Evaluate all initial configs
    for cfg in initial_configs:
        update_best(cfg)

    # Sort configs by initial score to prioritize promising ones
    config_scores = []
    for cfg in initial_configs:
        try:
            s = get_score(cfg)
            config_scores.append((s, cfg))
        except:
            config_scores.append((0.0, cfg))
    config_scores.sort(key=lambda x: -x[0])

    # === Optimize each configuration in normalized 22-param space ===
    for score_init, cfg in config_scores:
        if time_left() < 12.0:
            break

        x0 = points_to_params(cfg)

        # Progressive tau: start loose, tighten
        taus = [50, 100, 200, 400, 800, 1500]

        for tau in taus:
            if time_left() < 6.0:
                break

            # SLSQP
            try:
                res = minimize(smooth_objective, x0, args=(tau,), method='SLSQP',
                               options={'maxiter': 120, 'ftol': 1e-9})
                if res.fun < 1e5:
                    current_score = -res.fun
                    if current_score > best_score:
                        best_score = current_score
                        best_points = params_to_points(res.x)
                        x0 = res.x
            except:
                pass

            # L-BFGS-B for refinement
            if time_left() < 4.0:
                continue
            try:
                res2 = minimize(smooth_objective, x0, args=(tau,), method='L-BFGS-B',
                                options={'maxiter': 150, 'ftol': 1e-12})
                if res2.fun < 1e5:
                    current_score = -res2.fun
                    if current_score > best_score:
                        best_score = current_score
                        best_points = params_to_points(res2.x)
                        x0 = res2.x
            except:
                pass

        # Final high-tau refinement for this config
        if time_left() > 8.0:
            try:
                res_final = minimize(smooth_objective, x0, args=(2000,), method='L-BFGS-B',
                                     options={'maxiter': 100, 'ftol': 1e-13})
                if res_final.fun < 1e5:
                    current_score = -res_final.fun
                    if current_score > best_score:
                        best_score = current_score
                        best_points = params_to_points(res_final.x)
            except:
                pass

    # === Global Refinement on the best point found so far ===
    if best_points is not None and time_left() > 30.0:
        x_best = points_to_params(best_points)

        # Nelder-Mead with high tau
        try:
            res_nm = minimize(smooth_objective, x_best, args=(1000,), method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-11})
            if res_nm.fun < 1e5:
                pts = params_to_points(res_nm.x)
                update_best(pts)
        except:
            pass

    # === Powell refinement ===
    if best_points is not None and time_left() > 22.0:
        x_best = points_to_params(best_points)

        try:
            res_pw = minimize(smooth_objective, x_best, args=(1000,), method='Powell',
                             options={'maxiter': 1500, 'ftol': 1e-11, 'xtol': 1e-11})
            if res_pw.fun < 1e5:
                pts = params_to_points(res_pw.x)
                update_best(pts)
        except:
            pass

    # === Exact local search on active triangles ===
    if best_points is not None and time_left() > 15.0:
        def exact_objective(params):
            try:
                pts = params_to_points(params)
                areas = compute_all_areas(pts)
                min_area = np.min(areas)
                try:
                    hull = ConvexHull(pts)
                    hull_area = hull.volume
                    if hull_area < 1e-10:
                        return 1e6
                except:
                    return 1e6
                return -min_area / hull_area
            except:
                return 1e6

        x_best = points_to_params(best_points)

        # Nelder-Mead on exact (nonsmooth) objective
        try:
            res_exact = minimize(exact_objective, x_best, method='Nelder-Mead',
                                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res_exact.fun < 1e5:
                pts = params_to_points(res_exact.x)
                update_best(pts)
        except:
            pass

        # Powell on exact objective
        if time_left() > 10.0:
            try:
                res_pw_exact = minimize(exact_objective, x_best, method='Powell',
                                       options={'maxiter': 2000, 'ftol': 1e-12, 'xtol': 1e-12})
                if res_pw_exact.fun < 1e5:
                    pts = params_to_points(res_pw_exact.x)
                    update_best(pts)
            except:
                pass

    # === Local refinement with small perturbations around the best point ===
    if best_points is not None and time_left() > 10.0:
        x_best = points_to_params(best_points)

        for pert_scale in [0.003, 0.005, 0.01, 0.02]:
            if time_left() < 5.0:
                break
            for _ in range(3):
                if time_left() < 3.0:
                    break
                x_pert = x_best + rng.normal(0, pert_scale, len(x_best))
                try:
                    res_p = minimize(smooth_objective, x_pert, args=(2000,),
                                     method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-13})
                    if res_p.fun < 1e5:
                        pts = params_to_points(res_p.x)
                        update_best(pts)
                        if get_score(pts) > best_score:
                            x_best = res_p.x
                except:
                    pass

    # Final fallback: regular polygon
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = get_score(best_points)

    return best_points
