import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin for 150s limit

    # Precompute triangle indices (286 triangles)
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)

    def compute_score(points):
        """Compute min triangle area / convex hull area."""
        try:
            pts = points - points.mean(axis=0)
            hull = ConvexHull(pts)
            area_hull = hull.volume
        except Exception:
            return 0.0
        if area_hull < 1e-9:
            return 0.0

        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)
        return min_area / area_hull

    def _normalize_points(points):
        """Center and scale points to unit convex hull area."""
        pts = points - points.mean(axis=0)
        try:
            hull = ConvexHull(pts)
            area = hull.volume
            if area < 1e-9:
                return pts
            pts = pts * np.sqrt(1.0 / area)
        except Exception:
            pass
        return pts

    def _soft_objective(points_flat, beta):
        """
        Smooth (soft-min) objective for gradient-based optimization.
        Returns negative soft-min triangle area on unit hull area.
        Soft min is differentiable, so L-BFGS-B can use gradients.
        """
        pts = points_flat.reshape(-1, 2)
        pts = _normalize_points(pts)

        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross

        # Log-sum-exp trick for numerical stability
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta

        return -soft_min_area

    def _exact_objective(points_flat):
        """Exact (non-smooth) objective for final polish."""
        pts = points_flat.reshape(-1, 2)
        score = compute_score(pts)
        if score == 0.0:
            return 1e6
        return -score

    def _regular_polygon(n, radius=1.0, angle_offset=0.0):
        if n == 1:
            return np.array([[0.0, 0.0]])
        angles = np.linspace(angle_offset, 2 * np.pi + angle_offset, n, endpoint=False)
        xs = radius * np.cos(angles)
        ys = radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def _coordinate_descent(points, max_iter=200, tol=1e-11):
        """
        Coordinate descent: move one point at a time with adaptive step sizes.
        Exploits the structure that the min area is often determined by a few
        specific triangles, allowing targeted improvement.
        """
        pts = points.copy()
        best_score = compute_score(pts)

        # Start with larger steps, shrink as we converge
        step_scales = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]

        for iteration in range(max_iter):
            if time.time() > start_time + time_limit - 2:
                break

            # Use progressively smaller step sizes
            step_idx = min(iteration // 15, len(step_scales) - 1)
            base_step = step_scales[step_idx]

            improved = False
            for i in range(N):
                best_local = pts[i].copy()
                best_local_score = best_score

                directions = [
                    (base_step, 0), (-base_step, 0), (0, base_step), (0, -base_step),
                    (base_step * 0.707, base_step * 0.707), (-base_step * 0.707, base_step * 0.707),
                    (base_step * 0.707, -base_step * 0.707), (-base_step * 0.707, -base_step * 0.707),
                    (base_step * 0.866, base_step * 0.5), (-base_step * 0.866, base_step * 0.5),
                    (base_step * 0.866, -base_step * 0.5), (-base_step * 0.866, -base_step * 0.5),
                    # Smaller refinement steps
                    (base_step * 0.3, 0), (-base_step * 0.3, 0), (0, base_step * 0.3), (0, -base_step * 0.3),
                ]

                for dx, dy in directions:
                    new_pt = pts.copy()
                    new_pt[i] = pts[i] + np.array([dx, dy])
                    score = compute_score(new_pt)
                    if score > best_local_score + tol:
                        best_local_score = score
                        best_local = new_pt[i].copy()
                        improved = True

                if improved:
                    pts[i] = best_local
                    best_score = best_local_score

            if not improved:
                break

        return pts, best_score

    def _symmetric_objective(radii):
        """
        Objective for symmetric optimization: points lie on rays with fixed angles.
        radii: array of shape (N,)
        Returns negative min triangle area / hull area.
        """
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        xs = radii * np.cos(angles)
        ys = radii * np.sin(angles)
        pts = np.column_stack([xs, ys])
        
        # Normalize to unit hull area
        pts = _normalize_points(pts)
        
        score = compute_score(pts)
        if score == 0.0:
            return 1e6
        return -score

    # Phase 1: Symmetric Optimization
    # We assume 13-fold rotational symmetry initially.
    # Variables: radii r_0 ... r_12
    # Start with equal radii (regular polygon)
    init_radii = np.ones(N)
    
    best_symmetric_score = -float('inf')
    best_symmetric_radii = None

    # Use Nelder-Mead or Powell on the 13D space
    # Powell is often robust for non-smooth minima like min-area
    try:
        res_sym = opt.minimize(
            _symmetric_objective,
            init_radii,
            method='Powell',
            options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12}
        )
        if res_sym.success or res_sym.fun < 1e5:
            best_symmetric_radii = res_sym.x
            best_symmetric_score = -res_sym.fun
    except Exception:
        pass
    
    # Also try Nelder-Mead as a backup/complement
    try:
        res_nm = opt.minimize(
            _symmetric_objective,
            init_radii,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12}
        )
        if -res_nm.fun > best_symmetric_score:
            best_symmetric_score = -res_nm.fun
            best_symmetric_radii = res_nm.x
    except Exception:
        pass

    # Construct initial points from best symmetric solution
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    if best_symmetric_radii is not None:
        pts_x = best_symmetric_radii * np.cos(angles)
        pts_y = best_symmetric_radii * np.sin(angles)
        init_points = np.column_stack([pts_x, pts_y])
    else:
        init_points = _regular_polygon(N, 1.0, 0.0)
    
    # Normalize init_points for consistency in next stages
    init_points = _normalize_points(init_points)
    
    # Phase 2: Full-space optimization starting from symmetric solution
    # This allows symmetry breaking if it improves the score.
    
    best_score = best_symmetric_score
    best_points = init_points.copy()

    # Refine with L-BFGS-B using soft objective
    betas = [10.0, 50.0, 200.0, 1000.0]
    current = init_points.copy()
    for beta in betas:
        if time.time() > start_time + time_limit - 20:
            break
        try:
            res = opt.minimize(
                lambda x, b=beta: _soft_objective(x, b),
                current.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            current = res.x.reshape(-1, 2)
            score = compute_score(current)
            if score > best_score:
                best_score = score
                best_points = current.copy()
        except Exception:
            pass

    # Refine with Coordinate Descent
    if time.time() < start_time + time_limit - 10:
        try:
            current, score_cd = _coordinate_descent(best_points, max_iter=300)
            if score_cd > best_score:
                best_score = score_cd
                best_points = current.copy()
        except Exception:
            pass

    # Final polish with Nelder-Mead on exact objective
    if time.time() < start_time + time_limit - 5:
        try:
            res = opt.minimize(
                _exact_objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            candidate = res.x.reshape(-1, 2)
            score = compute_score(candidate)
            if score > best_score:
                best_score = score
                best_points = candidate.copy()
        except Exception:
            pass
    
    # Final coordinate descent pass
    if time.time() < start_time + time_limit - 2:
        try:
            candidate, score_cd2 = _coordinate_descent(best_points, max_iter=150)
            if score_cd2 > best_score:
                best_score = score_cd2
                best_points = candidate.copy()
        except Exception:
            pass

    return best_points
