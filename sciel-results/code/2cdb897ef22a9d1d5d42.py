import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull

N = 13

def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s margin for safety

    # Precompute indices for all triples
    tri_i = []
    tri_j = []
    tri_k = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
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
            return 0.0, 0.0
        if area_hull < 1e-9:
            return 0.0, 0.0
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
        return score, area_hull

    def _soft_min_score(points, beta=50.0):
        """Smooth surrogate using log-sum-exp for soft-min."""
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
        points = points_flat.reshape(-1, 2)
        return _soft_min_score(points, beta)

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
        if n == 1:
            return np.array([center])
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def _make_config(k_hull, k_inner, seed=0):
        """Create configuration with k_hull on regular polygon, k_inner inside."""
        rng = np.random.RandomState(seed)
        hull_pts = _regular_polygon(k_hull, radius=1.0)
        if k_inner == 0:
            return hull_pts
        if k_inner == 1:
            inner_pts = np.array([[0.0, 0.0]])
        elif k_inner == 2:
            inner_pts = np.array([[-0.3, 0.0], [0.3, 0.0]])
        else:
            inner_pts = _regular_polygon(k_inner, radius=0.35, center=(0.0, 0.0))
            inner_pts = inner_pts + rng.randn(k_inner, 2) * 0.02
        return np.vstack([hull_pts, inner_pts])

    best_points = None
    best_score = -float('inf')

    # Strategy: Focus on symmetric configurations with grid search over radial parameters
    # Key insight: enforce symmetry to avoid local minima, then optimize radially
    
    # Try various symmetric configurations
    # 1. All 13 on a circle (regular 13-gon)
    # 2. 7 outer + 6 inner (two concentric regular polygons)
    # 3. 6 outer + 1 center + 6 inner
    # 4. 5 outer + 8 inner
    # 5. 4 outer + 9 inner
    
    configurations = []
    
    # Regular 13-gon
    configurations.append(('13-gon', _regular_polygon(13, radius=1.0)))
    
    # Two-layer: 7 outer, 6 inner
    for r_inner in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        outer = _regular_polygon(7, radius=1.0)
        inner = _regular_polygon(6, radius=r_inner, center=(0.0, 0.0))
        configurations.append((f'7+6 r={r_inner}', np.vstack([outer, inner])))
    
    # Three-layer: 6 outer, 1 center, 6 inner
    for r_mid in [0.3, 0.4, 0.5, 0.6]:
        outer = _regular_polygon(6, radius=1.0)
        center_pt = np.array([[0.0, 0.0]])
        inner = _regular_polygon(6, radius=r_mid, center=(0.0, 0.0))
        configurations.append((f'6+1+6 r={r_mid}', np.vstack([outer, center_pt, inner])))
    
    # 5 outer + 8 inner
    for r_inner in [0.3, 0.4, 0.5]:
        outer = _regular_polygon(5, radius=1.0)
        inner = _regular_polygon(8, radius=r_inner, center=(0.0, 0.0))
        configurations.append((f'5+8 r={r_inner}', np.vstack([outer, inner])))
    
    # 4 outer + 9 inner
    for r_inner in [0.3, 0.4, 0.5]:
        outer = _regular_polygon(4, radius=1.0)
        inner = _regular_polygon(9, radius=r_inner, center=(0.0, 0.0))
        configurations.append((f'4+9 r={r_inner}', np.vstack([outer, inner])))
    
    # 8 outer + 5 inner
    for r_inner in [0.3, 0.4, 0.5]:
        outer = _regular_polygon(8, radius=1.0)
        inner = _regular_polygon(5, radius=r_inner, center=(0.0, 0.0))
        configurations.append((f'8+5 r={r_inner}', np.vstack([outer, inner])))
    
    # 9 outer + 4 inner
    for r_inner in [0.3, 0.4, 0.5]:
        outer = _regular_polygon(9, radius=1.0)
        inner = _regular_polygon(4, radius=r_inner, center=(0.0, 0.0))
        configurations.append((f'9+4 r={r_inner}', np.vstack([outer, inner])))
    
    # 10 outer + 3 inner
    for r_inner in [0.3, 0.4, 0.5]:
        outer = _regular_polygon(10, radius=1.0)
        inner = _regular_polygon(3, radius=r_inner, center=(0.0, 0.0))
        configurations.append((f'10+3 r={r_inner}', np.vstack([outer, inner])))
    
    # 11 outer + 2 inner
    for r_inner in [0.3, 0.4, 0.5]:
        outer = _regular_polygon(11, radius=1.0)
        inner = _regular_polygon(2, radius=r_inner, center=(0.0, 0.0))
        configurations.append((f'11+2 r={r_inner}', np.vstack([outer, inner])))
    
    # 12 outer + 1 inner
    for r_inner in [0.3, 0.4, 0.5]:
        outer = _regular_polygon(12, radius=1.0)
        inner = np.array([[r_inner, 0.0]])
        configurations.append((f'12+1 r={r_inner}', np.vstack([outer, inner])))
    
    # Evaluate all configurations
    for name, config in configurations:
        if time.time() > start_time + time_limit - 40:
            break
        s, _ = compute_score(config)
        if s > best_score:
            best_score = s
            best_points = config.copy()
    
    # Now optimize the best configurations found
    # Select top configurations for optimization
    config_scores = []
    for name, config in configurations:
        if time.time() > start_time + time_limit - 35:
            break
        s, _ = compute_score(config)
        config_scores.append((s, name, config))
    
    config_scores.sort(key=lambda x: -x[0])
    
    # Optimize top 5 configurations
    for rank, (s, name, config) in enumerate(config_scores[:5]):
        if time.time() > start_time + time_limit - 25:
            break
        if s < best_score * 0.95:  # Skip if much worse than current best
            continue
            
        current = config.copy()
        
        # Beta annealing optimization
        betas = [30.0, 80.0, 200.0, 500.0]
        for beta in betas:
            if time.time() > start_time + time_limit - 20:
                break
            try:
                result = opt.minimize(
                    lambda x, b=beta: _soft_objective(x, b),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 100}
                )
                optimized = result.x.reshape(-1, 2)
                opt_score, _ = compute_score(optimized)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = optimized.copy()
                current = optimized
            except Exception:
                pass
        
        # NM polish
        if best_points is not None and time.time() < start_time + time_limit - 15:
            try:
                res = opt.minimize(
                    _hard_score_neg,
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 150, 'adaptive': True, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                nm_points = res.x.reshape(-1, 2)
                nm_score, _ = compute_score(nm_points)
                if nm_score > best_score:
                    best_score = nm_score
                    best_points = nm_points.copy()
            except Exception:
                pass
    
    # Final high-precision polish
    if best_points is not None and time.time() < start_time + time_limit - 10:
        try:
            result = opt.minimize(
                lambda x: _soft_objective(x, 1000.0),
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 100}
            )
            polished = result.x.reshape(-1, 2)
            pol_score, _ = compute_score(polished)
            if pol_score > best_score:
                best_score = pol_score
                best_points = polished.copy()
        except Exception:
            pass
        
        # Final NM polish
        if time.time() < start_time + time_limit - 5:
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
