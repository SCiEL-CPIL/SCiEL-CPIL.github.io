import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull

N = 13

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s margin for safety

    # Precompute indices for all triples as numpy arrays for vectorized computation
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
    n_triples = len(tri_i)
    
    def compute_score(points):
        """Compute min triangle area / convex hull area for given points."""
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 0.0, 0.0
            
        if area_hull < 1e-9:
            return 0.0, 0.0

        x = points[:, 0]
        y = points[:, 1]
        # Vectorized triangle area computation
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)

        score = min_area / area_hull
        return score, area_hull

    def _soft_min_score(points, beta=50.0):
        """
        Smooth surrogate: use log-sum-exp to approximate the minimum triangle area.
        Returns negative soft-min score for minimization.
        """
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 1e6
            
        if area_hull < 1e-9:
            return 1e6

        x = points[:, 0]
        y = points[:, 1]
        
        # Vectorized triangle area computation
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross

        # Soft-min via log-sum-exp
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta

        score = soft_min_area / area_hull
        return -score

    def _hard_score_neg(points_flat):
        """Negative hard score for final evaluation."""
        points = points_flat.reshape(-1, 2)
        score, _ = compute_score(points)
        return -score

    def _soft_objective(points_flat, beta=50.0):
        """Soft-min objective for gradient-based optimization."""
        points = points_flat.reshape(-1, 2)
        return _soft_min_score(points, beta)

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
        """Generate points of a regular n-gon."""
        if n == 1:
            return np.array([center])
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def _make_symmetric_config(hull_n, inner_n, hull_r=1.0, inner_r=0.4, angle_offset=0.0, seed=0):
        """
        Create a configuration with hull_n points on a regular polygon
        and inner_n points arranged on a smaller concentric polygon with angle offset.
        """
        hull_pts = _regular_polygon(hull_n, radius=hull_r)
        
        if inner_n == 0:
            return hull_pts
        
        # Inner points: regular polygon with angle offset
        inner_angles = np.linspace(0, 2 * np.pi, inner_n, endpoint=False) + angle_offset
        inner_xs = inner_r * np.cos(inner_angles)
        inner_ys = inner_r * np.sin(inner_angles)
        inner_pts = np.column_stack([inner_xs, inner_ys])
        
        return np.vstack([hull_pts, inner_pts])

    best_points = None
    best_score = -float('inf')
    
    # === PHASE 1: Symmetric parameter grid search ===
    # For each (hull_n, inner_n) split, grid search over inner_r and angle_offset
    splits = [(7, 6), (8, 5), (9, 4), (6, 7), (10, 3), (11, 2), (12, 1), (13, 0), (5, 8)]
    
    for hull_n, inner_n in splits:
        if time.time() > start_time + time_limit - 40:
            break
        
        # Grid search over inner radius and angle offset
        inner_r_values = np.linspace(0.1, 0.7, 7)
        # Number of angle offsets depends on symmetry: 2*pi/inner_n * offsets
        n_offsets = 10
        angle_offsets = np.linspace(0, 2 * np.pi / inner_n, n_offsets, endpoint=False) if inner_n > 0 else [0.0]
        
        for ir in inner_r_values:
            if time.time() > start_time + time_limit - 35:
                break
            for ao in angle_offsets:
                if time.time() > start_time + time_limit - 35:
                    break
                config = _make_symmetric_config(hull_n, inner_n, 1.0, ir, ao)
                s, _ = compute_score(config)
                if s > best_score:
                    best_score = s
                    best_points = config.copy()
    
    # === PHASE 2: Optimize best symmetric config with L-BFGS-B ===
    if best_points is not None and time.time() < start_time + time_limit - 20:
        current = best_points.copy()
        for beta in [50.0, 200.0, 500.0, 1000.0]:
            if time.time() > start_time + time_limit - 15:
                break
            try:
                result = opt.minimize(
                    lambda x, b=beta: _soft_objective(x, b),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 100}
                )
                optimized = result.x.reshape(-1, 2)
                opt_score, _ = compute_score(optimized)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = optimized.copy()
                current = optimized
            except Exception:
                pass
    
    # === PHASE 3: Try multiple diverse starting points ===
    diverse_starts = [
        _make_symmetric_config(7, 6, 1.0, 0.35, 0.0),
        _make_symmetric_config(7, 6, 1.0, 0.5, np.pi/6),
        _make_symmetric_config(8, 5, 1.0, 0.3, 0.0),
        _make_symmetric_config(8, 5, 1.0, 0.5, np.pi/5),
        _make_symmetric_config(9, 4, 1.0, 0.4, 0.0),
        _make_symmetric_config(6, 7, 1.0, 0.35, 0.0),
    ]
    
    for config in diverse_starts:
        if time.time() > start_time + time_limit - 10:
            break
        s, _ = compute_score(config)
        if s > best_score:
            best_score = s
            best_points = config.copy()
        
        current = config.copy()
        for beta in [100.0, 400.0, 800.0]:
            if time.time() > start_time + time_limit - 8:
                break
            try:
                result = opt.minimize(
                    lambda x, b=beta: _soft_objective(x, b),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-13, 'maxls': 80}
                )
                optimized = result.x.reshape(-1, 2)
                opt_score, _ = compute_score(optimized)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = optimized.copy()
                current = optimized
            except Exception:
                pass
    
    # === PHASE 4: Final high-precision polish ===
    if best_points is not None and time.time() < start_time + time_limit - 5:
        try:
            result = opt.minimize(
                lambda x: _soft_objective(x, 1500.0),
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100}
            )
            polished = result.x.reshape(-1, 2)
            pol_score, _ = compute_score(polished)
            if pol_score > best_score:
                best_score = pol_score
                best_points = polished.copy()
        except Exception:
            pass
        
        if time.time() < start_time + time_limit - 2:
            try:
                res = opt.minimize(
                    _hard_score_neg,
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'adaptive': True, 'xatol': 1e-14, 'fatol': 1e-16}
                )
                final_points = res.x.reshape(-1, 2)
                final_score, _ = compute_score(final_points)
                if final_score > best_score:
                    best_score = final_score
                    best_points = final_points.copy()
            except Exception:
                pass

    if best_points is None:
        best_points = _regular_polygon(13, radius=1.0)
    
    return best_points

# EVOLVE-BLOCK-END
