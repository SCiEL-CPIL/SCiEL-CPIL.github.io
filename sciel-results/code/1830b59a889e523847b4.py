import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull

N = 13

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit

    # Precompute triangle indices
    tri_i = []
    tri_j = []
    tri_k = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                tri_i.append(i)
                tri_j.append(j)
                tri_k.append(k)
    tri_i = np.array(tri_i, dtype=np.int32)
    tri_j = np.array(tri_j, dtype=np.int32)
    tri_k = np.array(tri_k, dtype=np.int32)

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

    def _soft_objective(points_flat, beta=50.0):
        """
        Objective function for optimization.
        Returns negative soft-min triangle area (normalized to unit hull area).
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
        
        # Soft min
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta
        
        return -soft_min_area

    def _hard_objective(points_flat):
        """
        Objective function for final polish.
        Returns negative min triangle area (normalized to unit hull area).
        """
        pts = points_flat.reshape(-1, 2)
        pts = _normalize_points(pts)
        score = compute_score(pts)
        if score == 0.0:
            return 1e6
        return -score

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
        if n == 1:
            return np.array([center])
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    # === Symmetric parameterization: 1 center + 12 ring points ===
    # Parameters: [cx, cy, r_0, r_1, ..., r_11] (13 params)
    # Angles are fixed at 2*pi*k/12 for k=0..11
    
    ring_angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    ring_cos = np.cos(ring_angles)
    ring_sin = np.sin(ring_angles)

    def params_to_points(params):
        """Convert 13 params to 13 points. params = [cx, cy, r_0..r_11]"""
        cx, cy = params[0], params[1]
        radii = params[2:]
        # Ensure positive radii
        radii = np.maximum(radii, 0.01)
        x = cx + radii * ring_cos
        y = cy + radii * ring_sin
        pts = np.column_stack([x, y])
        # Add center point
        pts = np.vstack([pts, np.array([[cx, cy]])])
        return pts

    def symmetric_score(params):
        """Score for symmetric parameterization."""
        pts = params_to_points(params)
        return compute_score(pts)

    def symmetric_soft_objective(params, beta=50.0):
        """Soft objective for symmetric parameterization."""
        pts = params_to_points(params)
        pts = _normalize_points(pts)
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta
        
        return -soft_min_area

    def symmetric_hard_objective(params):
        """Hard objective for symmetric parameterization."""
        pts = params_to_points(params)
        pts = _normalize_points(pts)
        score = compute_score(pts)
        if score == 0.0:
            return 1e6
        return -score

    # Initial symmetric config: center at origin, all radii = 1.0
    base_params = np.zeros(13)
    base_params[2:] = 1.0

    best_score = -float('inf')
    best_points = None

    # Phase 1: Differential Evolution in reduced 13-dim symmetric space
    if time.time() < start_time + time_limit - 30:
        bounds = [(-1, 1), (-1, 1)] + [(0.1, 2.0)] * 12
        rng_seed = 42
        try:
            de_result = opt.differential_evolution(
                symmetric_hard_objective,
                bounds=bounds,
                seed=rng_seed,
                maxiter=50,
                popsize=15,
                tol=1e-8,
                polish=True,
                init='sobol',
                updating='immediate'
            )
            if de_result.success or de_result.fun < 0:
                sym_params = de_result.x
                pts = params_to_points(sym_params)
                score = compute_score(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
        except Exception:
            pass

    # Phase 2: Local polish in full 26-dim space starting from symmetric solution
    if best_points is not None and time.time() < start_time + time_limit - 15:
        current = best_points.copy()
        
        # L-BFGS-B with soft objective
        for beta in [50.0, 100.0, 200.0]:
            if time.time() > start_time + time_limit - 10:
                break
            try:
                res = opt.minimize(
                    lambda x, b=beta: _soft_objective(x, b),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                candidate = res.x.reshape(-1, 2)
                score = compute_score(candidate)
                if score > best_score:
                    best_score = score
                    best_points = candidate.copy()
                current = candidate
            except Exception:
                pass

        # Powell polish
        if time.time() < start_time + time_limit - 5:
            try:
                res = opt.minimize(
                    _hard_objective,
                    best_points.flatten(),
                    method='Powell',
                    options={'maxiter': 200, 'xtol': 1e-14, 'ftol': 1e-14}
                )
                candidate = res.x.reshape(-1, 2)
                score = compute_score(candidate)
                if score > best_score:
                    best_score = score
                    best_points = candidate.copy()
            except Exception:
                pass

    # Phase 3: Try a few diverse initializations in full space if time permits
    if time.time() < start_time + time_limit - 10:
        rng = np.random.RandomState(42)
        
        # 12 outer + 1 center with slight perturbations
        for pert_scale in [0.0, 0.05, 0.1, 0.15]:
            if time.time() > start_time + time_limit - 5:
                break
            outer = _regular_polygon(12, 1.0)
            center = np.array([[0.0, 0.0]])
            init = np.vstack([outer, center])
            if pert_scale > 0:
                init += rng.randn(*init.shape) * pert_scale
            
            current = init.copy()
            for beta in [50.0, 100.0]:
                if time.time() > start_time + time_limit - 3:
                    break
                try:
                    res = opt.minimize(
                        lambda x, b=beta: _soft_objective(x, b),
                        current.flatten(),
                        method='L-BFGS-B',
                        options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    current = res.x.reshape(-1, 2)
                except Exception:
                    pass
            
            if time.time() > start_time + time_limit - 2:
                break
            try:
                res = opt.minimize(
                    _hard_objective,
                    current.flatten(),
                    method='Powell',
                    options={'maxiter': 100, 'xtol': 1e-12, 'ftol': 1e-12}
                )
                candidate = res.x.reshape(-1, 2)
                score = compute_score(candidate)
                if score > best_score:
                    best_score = score
                    best_points = candidate.copy()
            except Exception:
                pass

    # Final fallback
    if best_points is None:
        best_points = params_to_points(base_params)

    return best_points
