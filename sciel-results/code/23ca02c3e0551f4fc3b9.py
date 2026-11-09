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
    deadline = start_time + 145.0
    N = 13
    best_score = 0.0
    best_points = None
    rng = np.random.RandomState(42)

    def time_left():
        return deadline - time.time()

    # === Normalization: fix p0=[0,0], p1=[1,0] -> 22 free params ===
    def params_to_points(params):
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(N - 2, 2)
        return pts

    def points_to_params(pts):
        """Normalize: translate p0 to origin, rotate p1 to x-axis, scale |p1-p0|=1."""
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
        """-soft_min(areas) / hull_area in normalized 22-param space."""
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

    def epigraph_objective(x, tau):
        """Epigraph formulation: minimize -t subject to t <= soft_min(areas)/hull_area.
        x = [params (22), t (1)] -> 23 variables"""
        try:
            params = x[:22]
            t = x[22]
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
            # Penalize if t > score
            penalty = 0.0
            if t > score:
                penalty = 1000.0 * (t - score) ** 2
            return -t + penalty
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

    # === Direct structural approach: build configurations from geometric principles ===
    def make_regular_ngon(n, center=None, radius=1.0, phase=0.0):
        angles = np.linspace(phase, phase + 2 * np.pi, n, endpoint=False)
        if center is None:
            center = np.array([0.0, 0.0])
        return np.column_stack((center[0] + radius * np.cos(angles),
                                center[1] + radius * np.sin(angles)))

    # The key insight: for maximizing min triangle area / hull area,
    # we want points spread as evenly as possible while keeping the hull compact.
    # The regular polygon is a natural starting point, but we can do better by
    # using a "nested" structure: outer ring + inner points placed at centers of
    # triangular faces of the outer polygon.

    initial_configs = []

    # 1. Regular 13-gon
    initial_configs.append(make_regular_ngon(13, radius=1.0))

    # 2. 12-gon ring + 1 center point
    ring12 = make_regular_ngon(12, radius=1.0)
    initial_configs.append(np.vstack([ring12, np.array([[0.0, 0.0]])]))

    # 3. 11-gon ring + 2 points (one center, one offset)
    ring11 = make_regular_ngon(11, radius=1.0)
    interior2 = np.array([[0.0, 0.0], [0.3, 0.0]])
    initial_configs.append(np.vstack([ring11, interior2]))

    # 4. 10-gon ring + 3 inner (triangle pattern)
    ring10 = make_regular_ngon(10, radius=1.0)
    inner3 = make_regular_ngon(3, radius=0.45, phase=np.pi/6)
    initial_configs.append(np.vstack([ring10, inner3]))

    # 5. 9-gon ring + 4 inner (square pattern)
    ring9 = make_regular_ngon(9, radius=1.0)
    inner4 = make_regular_ngon(4, radius=0.45, phase=np.pi/4)
    initial_configs.append(np.vstack([ring9, inner4]))

    # 6. 8-gon ring + 5 inner (pentagon pattern)
    ring8 = make_regular_ngon(8, radius=1.0)
    inner5 = make_regular_ngon(5, radius=0.5, phase=np.pi/5)
    initial_configs.append(np.vstack([ring8, inner5]))

    # 7. 7-gon ring + 6 inner (hexagon pattern)
    ring7 = make_regular_ngon(7, radius=1.0)
    inner6 = make_regular_ngon(6, radius=0.5, phase=0.0)
    initial_configs.append(np.vstack([ring7, inner6]))

    # 8. Two concentric rings: 7 outer + 6 inner
    outer7 = make_regular_ngon(7, radius=1.0)
    inner6 = make_regular_ngon(6, radius=0.55, phase=np.pi/6)
    initial_configs.append(np.vstack([outer7, inner6]))

    # 9. Three layers: 6 outer + 5 middle + 2 inner
    outer6 = make_regular_ngon(6, radius=1.0)
    middle5 = make_regular_ngon(5, radius=0.65, phase=np.pi/5)
    inner2 = np.array([[0.0, 0.0], [0.2, 0.0]])
    initial_configs.append(np.vstack([outer6, middle5, inner2]))

    # 10. Perturbed regular 13-gon
    base = make_regular_ngon(13, radius=1.0)
    perturbed = base + rng.normal(0, 0.03, base.shape)
    initial_configs.append(perturbed)

    # 11. Slightly elliptical 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    ellipse = np.column_stack((1.05 * np.cos(angles), 0.95 * np.sin(angles)))
    initial_configs.append(ellipse)

    # 12. 12-gon with center and one vertex slightly inward
    ring12c = make_regular_ngon(12, radius=1.0)
    ring12c[0] = ring12c[0] * 0.92
    initial_configs.append(np.vstack([ring12c, np.array([[0.0, 0.0]])]))

    # 13. 13 points on two circles (radial optimization)
    # Try different splits and radii
    for split in [6, 7, 8]:
        for r_inner in [0.4, 0.5, 0.6, 0.7]:
            n_outer = split
            n_inner = N - n_outer
            outer = make_regular_ngon(n_outer, radius=1.0)
            inner = make_regular_ngon(n_inner, radius=r_inner, phase=np.pi/n_inner)
            initial_configs.append(np.vstack([outer, inner]))

    # 14. Add more diverse configurations: 5 outer + 8 inner
    outer5 = make_regular_ngon(5, radius=1.0)
    inner8 = make_regular_ngon(8, radius=0.55, phase=np.pi/8)
    initial_configs.append(np.vstack([outer5, inner8]))

    # 15. 4 outer + 9 inner
    outer4 = make_regular_ngon(4, radius=1.0)
    inner9 = make_regular_ngon(9, radius=0.55, phase=np.pi/9)
    initial_configs.append(np.vstack([outer4, inner9]))

    # 16. Three rings: 5 + 4 + 4
    r1 = make_regular_ngon(5, radius=1.0)
    r2 = make_regular_ngon(4, radius=0.65, phase=np.pi/4)
    r3 = make_regular_ngon(4, radius=0.35, phase=np.pi/4 + np.pi/8)
    initial_configs.append(np.vstack([r1, r2, r3]))

    # 17. Three rings: 6 + 4 + 3
    r1 = make_regular_ngon(6, radius=1.0)
    r2 = make_regular_ngon(4, radius=0.6, phase=np.pi/4)
    r3 = make_regular_ngon(3, radius=0.3, phase=0.0)
    initial_configs.append(np.vstack([r1, r2, r3]))

    # 18. Three rings: 4 + 5 + 4
    r1 = make_regular_ngon(4, radius=1.0)
    r2 = make_regular_ngon(5, radius=0.65, phase=np.pi/5)
    r3 = make_regular_ngon(4, radius=0.35, phase=np.pi/4)
    initial_configs.append(np.vstack([r1, r2, r3]))

    # 19. Golden ratio spiral
    n = N
    theta = np.linspace(0, 4 * np.pi, n)
    r = 0.9 * np.sqrt(np.linspace(0.1, 1.0, n))
    spiral = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    initial_configs.append(spiral)

    # 20. Fibonacci spiral
    phi = (1 + np.sqrt(5)) / 2
    theta = np.arange(n) * 2 * np.pi / phi
    r = 0.9 * np.sqrt(np.linspace(0.05, 1.0, n))
    fib_spiral = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    initial_configs.append(fib_spiral)

    # Evaluate all initial configs
    for cfg in initial_configs:
        update_best(cfg)

    # === Optimize each configuration in normalized 22-param space ===
    # Prioritize configs by their initial score
    config_scores = []
    for cfg in initial_configs:
        s = get_score(cfg)
        config_scores.append((s, cfg))
    config_scores.sort(key=lambda x: -x[0])

    # Limit number of configs to optimize to save time
    max_configs = 12
    configs_to_optimize = config_scores[:max_configs]

    for score_val, cfg in configs_to_optimize:
        if time_left() < 20.0:
            break

        x0 = points_to_params(cfg)

        # Progressive tau: start loose, tighten
        taus = [50, 100, 200, 300, 400]

        for tau in taus:
            if time_left() < 12.0:
                break

            # SLSQP (uses gradient)
            try:
                res = minimize(smooth_objective, x0, args=(tau,), method='SLSQP',
                               options={'maxiter': 300, 'ftol': 1e-10})
                if res.fun < 1e5:
                    current_score = -res.fun
                    if current_score > best_score:
                        best_score = current_score
                        best_points = params_to_points(res.x)
                        x0 = res.x
            except:
                pass

            # L-BFGS-B for refinement
            if time_left() < 8.0:
                continue
            try:
                res2 = minimize(smooth_objective, x0, args=(tau,), method='L-BFGS-B',
                                options={'maxiter': 200, 'ftol': 1e-12})
                if res2.fun < 1e5:
                    current_score = -res2.fun
                    if current_score > best_score:
                        best_score = current_score
                        best_points = params_to_points(res2.x)
                        x0 = res2.x
            except:
                pass

        # Final high-tau refinement for this config
        if time_left() > 15.0:
            try:
                res_final = minimize(smooth_objective, x0, args=(500,), method='L-BFGS-B',
                                     options={'maxiter': 150, 'ftol': 1e-13})
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
        
        # Nelder-Mead with high tau for sharpness
        try:
            res_nm = minimize(smooth_objective, x_best, args=(500,), method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-11, 'fatol': 1e-11})
            if res_nm.fun < 1e5:
                pts = params_to_points(res_nm.x)
                update_best(pts)
        except:
            pass

    # === Powell refinement ===
    if best_points is not None and time_left() > 25.0:
        x_best = points_to_params(best_points)
        
        try:
            res_pw = minimize(smooth_objective, x_best, args=(500,), method='Powell',
                             options={'maxiter': 2000, 'ftol': 1e-11, 'xtol': 1e-11})
            if res_pw.fun < 1e5:
                pts = params_to_points(res_pw.x)
                update_best(pts)
        except:
            pass

    # === Epigraph formulation refinement ===
    if best_points is not None and time_left() > 20.0:
        x_best = points_to_params(best_points)
        # Initialize t with current score
        t_init = best_score
        x_epigraph = np.concatenate([x_best, [t_init]])
        
        try:
            res_epi = minimize(epigraph_objective, x_epigraph, args=(500,), method='L-BFGS-B',
                              options={'maxiter': 200, 'ftol': 1e-12})
            if res_epi.fun < 1e5:
                pts = params_to_points(res_epi.x[:22])
                update_best(pts)
        except:
            pass

    # === Local refinement with small perturbations around the best point ===
    if best_points is not None and time_left() > 25.0:
        x_best = points_to_params(best_points)

        # Try several perturbation scales
        for pert_scale in [0.002, 0.005, 0.01, 0.02, 0.03]:
            if time_left() < 12.0:
                break
            for _ in range(8):
                if time_left() < 6.0:
                    break
                x_pert = x_best + rng.normal(0, pert_scale, len(x_best))
                try:
                    res_p = minimize(smooth_objective, x_pert, args=(500,),
                                     method='L-BFGS-B', options={'maxiter': 150, 'ftol': 1e-13})
                    if res_p.fun < 1e5:
                        pts = params_to_points(res_p.x)
                        update_best(pts)
                        if get_score(pts) > best_score:
                            x_best = res_p.x
                except:
                    pass

    # === Final high-precision polish pass ===
    if best_points is not None and time_left() > 15.0:
        x_best = points_to_params(best_points)
        
        # Very tight tolerance L-BFGS-B
        try:
            res_polish = minimize(smooth_objective, x_best, args=(600,), method='L-BFGS-B',
                                  options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            if res_polish.fun < 1e5:
                pts = params_to_points(res_polish.x)
                update_best(pts)
        except:
            pass
        
        # SLSQP with very tight tolerance
        if time_left() > 10.0:
            try:
                res_polish2 = minimize(smooth_objective, x_best, args=(600,), method='SLSQP',
                                       options={'maxiter': 200, 'ftol': 1e-14})
                if res_polish2.fun < 1e5:
                    pts = params_to_points(res_polish2.x)
                    update_best(pts)
            except:
                pass

    # Final fallback: regular polygon
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = get_score(best_points)

    return best_points
