import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay

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
    """Soft minimum with log-sum-exp stability."""
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

    # === Generate diverse initial configurations ===
    def make_regular_ngon(n, center=None, radius=1.0, phase=0.0):
        angles = np.linspace(phase, phase + 2 * np.pi, n, endpoint=False)
        if center is None:
            center = np.array([0.0, 0.0])
        return np.column_stack((center[0] + radius * np.cos(angles),
                                center[1] + radius * np.sin(angles)))

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

    # 14. 3x5 Grid
    grid35 = np.array([
        [0, 0], [1, 0], [2, 0], [3, 0], [4, 0],
        [0, 1], [1, 1], [2, 1], [3, 1], [4, 1],
        [0, 2], [1, 2], [2, 2], [3, 2], [4, 2]
    ])
    grid35_pts = grid35[:13] - np.mean(grid35[:13], axis=0)
    scale = 1.0 / np.max(np.linalg.norm(grid35_pts, axis=1))
    initial_configs.append(grid35_pts * scale)

    # 15. Hexagonal packing subset
    hex_pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i + 0.5 * j
            y = np.sqrt(3)/2 * j
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    radii = np.linalg.norm(hex_pts, axis=1)
    sorted_idx = np.argsort(radii)
    hex_subset = hex_pts[sorted_idx[:13]]
    hex_subset -= np.mean(hex_subset, axis=0)
    scale = 1.0 / np.max(np.linalg.norm(hex_subset, axis=1))
    initial_configs.append(hex_subset * scale)

    # 16. Delaunay-based initial config: start from regular polygon, perturb to break symmetry
    base13 = make_regular_ngon(13, radius=1.0)
    # Perturb vertices radially and angularly to create a non-regular mesh
    theta = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    r = np.ones(13)
    r[::3] = 1.05  # every 3rd vertex slightly outward
    r[1::3] = 0.95  # every 3rd vertex starting at 1 slightly inward
    delaunay_init = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    # Add a few interior points
    interior_pts = make_regular_ngon(5, radius=0.4, phase=np.pi/5)
    delaunay_init_full = np.vstack([delaunay_init[:8], interior_pts])
    initial_configs.append(delaunay_init_full)

    # 17. Another Delaunay-inspired: 6 outer + 7 inner in two rings
    outer6 = make_regular_ngon(6, radius=1.0)
    mid_ring = make_regular_ngon(4, radius=0.6, phase=np.pi/4)
    inner3 = make_regular_ngon(3, radius=0.3, phase=0)
    initial_configs.append(np.vstack([outer6, mid_ring, inner3]))

    # 18. Perturbed hexagonal packing
    hex_init = initial_configs[14].copy()
    hex_init += rng.normal(0, 0.03, hex_init.shape)
    initial_configs.append(hex_init)

    # 19. Random with Delaunay-like structure
    # Start with a convex hull of 7 points, then add 6 interior points
    hull_pts = make_regular_ngon(7, radius=1.0) + rng.normal(0, 0.02, (7, 2))
    interior_rng = rng.uniform(0.2, 0.7, (6, 2))
    interior_angles = rng.uniform(0, 2 * np.pi, 6)
    interior_pts2 = np.column_stack((interior_rng[:, 0] * np.cos(interior_angles),
                                      interior_rng[:, 1] * np.sin(interior_angles)))
    initial_configs.append(np.vstack([hull_pts, interior_pts2]))

    # 20. Square with structured interior
    sq_outer = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1],
                         [1, 0], [0, 1], [-1, 0], [0, -1]])
    sq_inner = make_regular_ngon(5, radius=0.45, phase=np.pi/10)
    initial_configs.append(np.vstack([sq_outer, sq_inner]))

    # Evaluate all initial configs
    for cfg in initial_configs:
        update_best(cfg)

    # === Optimize each configuration in normalized 22-param space ===
    for cfg_idx, cfg in enumerate(initial_configs):
        if time_left() < 15.0:
            break

        x0 = points_to_params(cfg)

        # Progressive tau: start loose, tighten
        taus = [50, 100, 200, 400, 800]

        for tau in taus:
            if time_left() < 8.0:
                break

            # SLSQP (uses gradient)
            try:
                res = minimize(smooth_objective, x0, args=(tau,), method='SLSQP',
                               options={'maxiter': 150, 'ftol': 1e-10})
                if res.fun < 1e5:
                    current_score = -res.fun
                    if current_score > best_score:
                        best_score = current_score
                        best_points = params_to_points(res.x)
                        x0 = res.x
            except:
                pass

            # L-BFGS-B for refinement
            if time_left() < 5.0:
                continue
            try:
                res2 = minimize(smooth_objective, x0, args=(tau,), method='L-BFGS-B',
                                options={'maxiter': 100, 'ftol': 1e-12})
                if res2.fun < 1e5:
                    current_score = -res2.fun
                    if current_score > best_score:
                        best_score = current_score
                        best_points = params_to_points(res2.x)
                        x0 = res2.x
            except:
                pass

    # === Global Refinement on the best point found so far ===
    if best_points is not None and time_left() > 25.0:
        x_best = points_to_params(best_points)
        
        # High-tau L-BFGS-B refinement
        try:
            res_hi = minimize(smooth_objective, x_best, args=(1000,), method='L-BFGS-B',
                             options={'maxiter': 200, 'ftol': 1e-13})
            if res_hi.fun < 1e5:
                pts = params_to_points(res_hi.x)
                update_best(pts)
                x_best = res_hi.x
        except:
            pass

        # Nelder-Mead with high tau for sharpness
        if time_left() > 15.0:
            try:
                res_nm = minimize(smooth_objective, x_best, args=(1000,), method='Nelder-Mead',
                                 options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
                if res_nm.fun < 1e5:
                    pts = params_to_points(res_nm.x)
                    update_best(pts)
            except:
                pass

    # === Powell refinement ===
    if best_points is not None and time_left() > 15.0:
        x_best = points_to_params(best_points)
        
        try:
            res_pw = minimize(smooth_objective, x_best, args=(1000,), method='Powell',
                             options={'maxiter': 1000, 'ftol': 1e-11, 'xtol': 1e-11})
            if res_pw.fun < 1e5:
                pts = params_to_points(res_pw.x)
                update_best(pts)
        except:
            pass

    # === Local refinement with small perturbations around the best point ===
    if best_points is not None and time_left() > 15.0:
        x_best = points_to_params(best_points)

        # Try several perturbation scales
        for pert_scale in [0.003, 0.008, 0.015]:
            if time_left() < 8.0:
                break
            for _ in range(4):
                if time_left() < 5.0:
                    break
                x_pert = x_best + rng.normal(0, pert_scale, len(x_best))
                try:
                    res_p = minimize(smooth_objective, x_pert, args=(1000,),
                                     method='L-BFGS-B', options={'maxiter': 80, 'ftol': 1e-12})
                    if res_p.fun < 1e5:
                        pts = params_to_points(res_p.x)
                        s = get_score(pts)
                        if s > best_score:
                            best_score = s
                            best_points = pts.copy()
                            x_best = res_p.x
                except:
                    pass

    # Final fallback: regular polygon
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = get_score(best_points)

    return best_points
