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
    # Clip to avoid overflow in exp
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
    deadline = start_time + 140.0  # Safety margin before 150s limit
    N = 13
    best_score = 0.0
    best_points = None

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

    # 3. Regular 11-gon + 2 interior (symmetric pair)
    ring11 = make_regular_ngon(11, radius=1.0)
    interior2 = np.array([[0.35, 0.0], [-0.35, 0.0]])
    initial_configs.append(np.vstack([ring11, interior2]))

    # 4. 7 outer + 6 inner concentric
    outer7 = make_regular_ngon(7, radius=1.0)
    inner6 = make_regular_ngon(6, radius=0.55, phase=np.pi/6)
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
    inner5 = make_regular_ngon(5, radius=0.5, phase=np.pi/5)
    initial_configs.append(np.vstack([outer8, inner5]))

    # 8. 12 outer + 1 center with slight offset (breaks symmetry to help optimization)
    ring12b = make_regular_ngon(12, radius=1.0, phase=np.pi/12)
    initial_configs.append(np.vstack([ring12b, np.array([[0.0, 0.0]])]))

    # 9. 11 outer + 2 inner (radial pair at different angles)
    ring11b = make_regular_ngon(11, radius=1.0, phase=np.pi/11)
    interior2b = np.array([[0.4, 0.0], [0.2, 0.2]])
    initial_configs.append(np.vstack([ring11b, interior2b]))

    # Evaluate all initial configs
    for cfg in initial_configs:
        s = get_score(cfg)
        if s > best_score:
            best_score = s
            best_points = cfg.copy()

    # === Optimize each configuration in normalized 22-param space ===
    for cfg_idx, cfg in enumerate(initial_configs):
        if time_left() < 12.0:
            break

        x0 = points_to_params(cfg)

        # Progressive tau: start loose, tighten more aggressively
        taus = [30, 60, 100, 180, 300]

        for tau in taus:
            if time_left() < 8.0:
                break

            # SLSQP (uses gradient)
            try:
                res = minimize(smooth_objective, x0, args=(tau,), method='SLSQP',
                               options={'maxiter': 400, 'ftol': 1e-10})
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
                                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
                if res2.fun < 1e5:
                    current_score = -res2.fun
                    if current_score > best_score:
                        best_score = current_score
                        best_points = params_to_points(res2.x)
                        x0 = res2.x
            except:
                pass

        # Final high-tau refinement
        if time_left() > 10.0:
            try:
                res_final = minimize(smooth_objective, x0, args=(400,), method='L-BFGS-B',
                                     options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-9})
                if res_final.fun < 1e5:
                    current_score = -res_final.fun
                    if current_score > best_score:
                        best_score = current_score
                        best_points = params_to_points(res_final.x)
            except:
                pass

    # === Full 2D refinement on the best point set (unconstrained space) ===
    if time_left() > 25.0 and best_points is not None:
        def neg_score_2d(pts_flat, tau):
            pts = pts_flat.reshape(N, 2)
            try:
                areas = compute_all_areas(pts)
                sm = soft_min(areas, tau)
                hull = ConvexHull(pts)
                hull_area = hull.volume
                if hull_area < 1e-10:
                    return 1e6
                return -(sm / hull_area)
            except:
                return 1e6

        x0_2d = best_points.flatten()
        for tau in [150, 250]:
            if time_left() < 12.0:
                break
            for r in range(3):
                if time_left() < 6.0:
                    break
                if r == 0:
                    x_start = x0_2d.copy()
                else:
                    x_start = x0_2d + np.random.normal(0, 0.02, 2 * N)
                try:
                    res2d = minimize(neg_score_2d, x_start, args=(tau,), method='Powell',
                                     options={'maxiter': 4000, 'xatol': 1e-7, 'fatol': 1e-10})
                    if res2d.fun < 1e5:
                        current_score = -res2d.fun
                        if current_score > best_score:
                            best_score = current_score
                            best_points = res2d.x.reshape(N, 2)
                            x0_2d = res2d.x
                except:
                    pass

    # === Final exact Nelder-Mead polish on the true nonsmooth objective ===
    if time_left() > 15.0 and best_points is not None:
        def exact_neg_score(pts_flat):
            pts = pts_flat.reshape(N, 2)
            try:
                min_area = compute_min_triangle_area(pts)
                hull = ConvexHull(pts)
                hull_area = hull.volume
                if hull_area < 1e-10:
                    return 1e6
                return -(min_area / hull_area)
            except:
                return 1e6

        x0_nm = best_points.flatten()
        # Try a few small perturbations to escape local minima of the nonsmooth landscape
        for pert_scale in [0.0, 0.01, 0.02]:
            if time_left() < 5.0:
                break
            if pert_scale == 0.0:
                x_start = x0_nm.copy()
            else:
                x_start = x0_nm + np.random.normal(0, pert_scale, 2 * N)
            try:
                res_nm = minimize(exact_neg_score, x_start, method='Nelder-Mead',
                                  options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10,
                                           'adaptive': True})
                if res_nm.fun < 1e5:
                    current_score = -res_nm.fun
                    if current_score > best_score:
                        best_score = current_score
                        best_points = res_nm.x.reshape(N, 2)
                        x0_nm = res_nm.x
            except:
                pass

    # Final exact evaluation
    if best_points is not None:
        final_score = get_score(best_points)
        if final_score > best_score:
            best_score = final_score

    return best_points
