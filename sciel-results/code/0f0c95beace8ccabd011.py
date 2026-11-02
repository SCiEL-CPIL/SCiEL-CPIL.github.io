import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
import itertools

N = 13

def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s margin for safety

    # Precompute indices for all triples
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
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 0.0
            
        if area_hull < 1e-9:
            return 0.0

        x = points[:, 0]
        y = points[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)

        return min_area / area_hull

    def _neg_score(points_flat):
        """Negative score for minimization."""
        points = points_flat.reshape(-1, 2)
        return -compute_score(points)
    
    def _soft_min_score(points_flat, beta=50.0):
        """Soft minimum of triangle areas, normalized by hull area."""
        points = points_flat.reshape(-1, 2)
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 1e6
            
        if area_hull < 1e-9:
            return 1e6

        x = points[:, 0]
        y = points[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)
        score = min_area / area_hull
        
        # Soft min is a smooth approximation to min
        # We want to maximize min area, so minimize -soft_min
        # Using log-sum-exp trick for stability
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min = -log_sum_exp / beta
        
        return -soft_min / area_hull

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    best_points = None
    best_score = -float('inf')
    
    # Generate diverse initial configurations
    configs = []
    
    # 1. Regular 13-gon
    configs.append(_regular_polygon(13, radius=1.0))
    
    # 2. Regular 13-gon with small random perturbations (break symmetry)
    for seed in [1, 2, 3, 4, 5]:
        rng = np.random.RandomState(seed)
        p = _regular_polygon(13, radius=1.0)
        p = p + rng.randn(13, 2) * 0.01
        configs.append(p)
        
    # 3. Ring configurations
    # 1 center + 12-gon
    center = np.array([[0.0, 0.0]])
    outer_12 = _regular_polygon(12, radius=1.0)
    configs.append(np.vstack([center, outer_12]))
    
    # 2 centers + 11-gon
    c1 = np.array([[-0.1, 0.0]])
    c2 = np.array([[0.1, 0.0]])
    outer_11 = _regular_polygon(11, radius=1.0)
    configs.append(np.vstack([c1, c2, outer_11]))
    
    # 4. Jittered Grid (4x4 grid, remove 3 points to get 13)
    grid_x = np.linspace(-1, 1, 4)
    grid_y = np.linspace(-1, 1, 4)
    gx, gy = np.meshgrid(grid_x, grid_y)
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()])
    rng = np.random.RandomState(42)
    indices = rng.permutation(16)
    configs.append(grid_pts[indices[:13]])
    
    # 5. Random points in unit disk
    for seed in [10, 11, 12]:
        rng = np.random.RandomState(seed)
        angles = rng.uniform(0, 2*np.pi, 13)
        radii = np.sqrt(rng.uniform(0, 1, 13))
        xs = radii * np.cos(angles)
        ys = radii * np.sin(angles)
        configs.append(np.column_stack([xs, ys]))

    # Optimize each configuration
    for i, config in enumerate(configs):
        if time.time() > start_time + time_limit - 20:
            break
            
        # Evaluate initial score
        s = compute_score(config)
        if s > best_score:
            best_score = s
            best_points = config.copy()
        
        # First pass: Soft objective to smooth the landscape (L-BFGS-B)
        # This helps escape local minima of the hard min function
        try:
            result_soft = opt.minimize(
                lambda x: _soft_min_score(x, beta=50.0),
                config.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            soft_points = result_soft.x.reshape(-1, 2)
            soft_score = compute_score(soft_points)
            if soft_score > best_score:
                best_score = soft_score
                best_points = soft_points.copy()
        except Exception:
            pass

        # Second pass: Hard objective with Nelder-Mead (robust local search)
        # Start from the best of the initial and soft-optimized points
        current = best_points.copy()
        if compute_score(config) > compute_score(current):
            current = config.copy()

        try:
            result = opt.minimize(
                _neg_score,
                current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'adaptive': True, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            opt_points = result.x.reshape(-1, 2)
            opt_score = compute_score(opt_points)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
        except Exception:
            pass

    # Final polish on the best found
    if best_points is not None and time.time() < start_time + time_limit - 5:
        # Try Powell's method for the final polish, which is often better for
        # non-smooth functions like min-area
        try:
            result = opt.minimize(
                _neg_score,
                best_points.flatten(),
                method='Powell',
                options={'maxiter': 200, 'xtol': 1e-12, 'ftol': 1e-12}
            )
            final_points = result.x.reshape(-1, 2)
            final_score = compute_score(final_points)
            if final_score > best_score:
                best_score = final_score
                best_points = final_points.copy()
        except Exception:
            pass
        
        # One more Nelder-Mead pass if time permits
        if time.time() < start_time + time_limit - 2:
            try:
                result = opt.minimize(
                    _neg_score,
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'adaptive': True, 'xatol': 1e-14, 'fatol': 1e-16}
                )
                final_points = result.x.reshape(-1, 2)
                final_score = compute_score(final_points)
                if final_score > best_score:
                    best_score = final_score
                    best_points = final_points.copy()
            except Exception:
                pass

    if best_points is None:
        best_points = _regular_polygon(13, radius=1.0)
        
    return best_points
