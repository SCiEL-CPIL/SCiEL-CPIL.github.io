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

        # Soft-min via log-sum-exp (more stable formulation)
        min_area = np.min(areas)
        # Subtract min for numerical stability: exp(-beta*(area - min_area)) is well-conditioned
        shifted = -beta * (areas - min_area)
        max_shifted = np.max(shifted)
        log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
        soft_min_area = min_area - log_sum_exp / beta

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

    def _make_config(k_hull, k_inner, seed=0):
        """
        Create a configuration with k_hull points on a regular polygon
        and k_inner points arranged inside.
        """
        rng = np.random.RandomState(seed)
        hull_pts = _regular_polygon(k_hull, radius=1.0)
        
        if k_inner == 0:
            return hull_pts
        
        # Arrange interior points
        if k_inner == 1:
            inner_pts = np.array([[0.0, 0.0]])
        elif k_inner == 2:
            # Place two points along a diameter
            inner_pts = np.array([[-0.3, 0.0], [0.3, 0.0]])
        else:
            # Place interior points on a smaller circle
            inner_pts = _regular_polygon(k_inner, radius=0.35, center=(0.0, 0.0))
            # Add small random perturbation to break symmetry
            inner_pts = inner_pts + rng.randn(k_inner, 2) * 0.02
        
        return np.vstack([hull_pts, inner_pts])

    best_points = None
    best_score = -float('inf')
    
    # Focus on promising k_hull values that tend to perform well
    promising_k_hull = [3, 4, 5, 6, 7, 8, 10, 13]
    
    for k_hull in promising_k_hull:
        if time.time() > start_time + time_limit - 30:
            break
        
        k_inner = N - k_hull
        config = _make_config(k_hull, k_inner, seed=k_hull)
        
        # Evaluate initial config
        s, _ = compute_score(config)
        if s > best_score:
            best_score = s
            best_points = config.copy()
        
        # Adaptive beta annealing: start moderate, increase as we converge
        current = config.copy()
        for beta in [30.0, 80.0, 200.0, 500.0, 1000.0]:
            if time.time() > start_time + time_limit - 25:
                break
            try:
                result = opt.minimize(
                    lambda x, b=beta: _soft_objective(x, b),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 120}
                )
                optimized = result.x.reshape(-1, 2)
                opt_score, _ = compute_score(optimized)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = optimized.copy()
                current = optimized
            except Exception:
                pass
        
        # Quick NM polish on best found so far
        if best_points is not None and time.time() < start_time + time_limit - 20:
            try:
                res = opt.minimize(
                    _hard_score_neg,
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'adaptive': True, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                nm_points = res.x.reshape(-1, 2)
                nm_score, _ = compute_score(nm_points)
                if nm_score > best_score:
                    best_score = nm_score
                    best_points = nm_points.copy()
            except Exception:
                pass

    # Additional: try perturbed 13-gon configs for diversity
    if time.time() < start_time + time_limit - 35:
        for seed in [0, 1, 2, 3, 4, 5]:
            if time.time() > start_time + time_limit - 30:
                break
            
            rng = np.random.RandomState(seed + 100)
            config = _regular_polygon(N, radius=1.0)
            config = config + rng.randn(N, 2) * 0.02
            
            s, _ = compute_score(config)
            if s > best_score:
                best_score = s
                best_points = config.copy()
            
            current = config.copy()
            for beta in [50.0, 150.0, 400.0, 800.0]:
                if time.time() > start_time + time_limit - 25:
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

    # Final global polish with high beta
    if best_points is not None and time.time() < start_time + time_limit - 10:
        try:
            result = opt.minimize(
                lambda x: _soft_objective(x, 1200.0),
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-17, 'gtol': 1e-16, 'maxls': 150}
            )
            polished = result.x.reshape(-1, 2)
            pol_score, _ = compute_score(polished)
            if pol_score > best_score:
                best_score = pol_score
                best_points = polished.copy()
        except Exception:
            pass
        
        if time.time() < start_time + time_limit - 5:
            try:
                res = opt.minimize(
                    _hard_score_neg,
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 800, 'adaptive': True, 'xatol': 1e-15, 'fatol': 1e-17}
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
