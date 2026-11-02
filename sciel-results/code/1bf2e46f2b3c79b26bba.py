"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import time


def _triangle_areas(pts):
    """Compute all triangle areas for N points using vectorized operations."""
    n = len(pts)
    # For each pair (i,j), compute cross product with all k > j
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            # Vector from i to j
            v1 = pts[j] - pts[i]
            # Vectors from i to all k > j
            if j + 1 < n:
                v2s = pts[j + 1:] - pts[i]
                # Cross product z-component: v1_x * v2_y - v1_y * v2_x
                crosses = v1[0] * v2s[:, 1] - v1[1] * v2s[:, 0]
                areas.extend(np.abs(crosses) * 0.5)
    return np.array(areas) if areas else np.array([0.0])


def _score_points(pts):
    """Compute min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area <= 0:
            return 0.0
    except Exception:
        return 0.0
    
    # Compute all triangle areas
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    
    min_area = np.min(areas)
    return min_area / hull_area


def _regular_polygon_points(n, radius=1.0):
    """Generate n points on a regular polygon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.column_stack([x, y])


def _flatten_params(pts):
    """Flatten 2D points to 1D parameter vector."""
    return pts.flatten()


def _unflatten_params(params):
    """Unflatten 1D parameter vector to 2D points."""
    n = len(params) // 2
    return params.reshape(n, 2)


def _neg_score(params):
    """Negative score for minimization."""
    pts = _unflatten_params(params)
    return -_score_points(pts)


def construct():
    """Construct 13 points to maximize minimum triangle area / convex hull area."""
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Start with regular polygon
    best_pts = _regular_polygon_points(N, radius=1.0)
    best_score = _score_points(best_pts)
    
    # Try different initial configurations
    initial_configs = [
        _regular_polygon_points(N, radius=1.0),
        _regular_polygon_points(N, radius=1.5),
        _regular_polygon_points(N, radius=0.5),
    ]
    
    # Add some random perturbations
    rng = np.random.RandomState(42)
    for _ in range(5):
        base = _regular_polygon_points(N, radius=1.0)
        perturbed = base + rng.normal(0, 0.1, size=base.shape)
        initial_configs.append(perturbed)
    
    # Optimize each configuration
    for config in initial_configs:
        if time.time() > deadline:
            break
        
        params = _flatten_params(config)
        
        # Bounds: keep points in a reasonable range
        bounds = [(-3, 3)] * (2 * N)
        
        # Use differential evolution for global optimization
        try:
            result = differential_evolution(
                _neg_score,
                bounds,
                maxiter=50,
                tol=1e-6,
                seed=42,
                polish=False,
                workers=1
            )
            
            if result.success or result.fun < -best_score:
                candidate_pts = _unflatten_params(result.x)
                candidate_score = _score_points(candidate_pts)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_pts = candidate_pts.copy()
        except Exception:
            continue
        
        # Local refinement with L-BFGS-B
        if time.time() > deadline:
            break
        
        try:
            local_result = minimize(
                _neg_score,
                result.x if 'result' in dir() else params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            
            if local_result.fun < -best_score:
                candidate_pts = _unflatten_params(local_result.x)
                candidate_score = _score_points(candidate_pts)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_pts = candidate_pts.copy()
        except Exception:
            continue
    
    # Final check: ensure we're returning a valid result
    if time.time() > deadline - 5:
        # Return best found so far
        return best_pts
    
    # One more refinement pass if time permits
    try:
        params = _flatten_params(best_pts)
        bounds = [(-3, 3)] * (2 * N)
        
        final_result = minimize(
            _neg_score,
            params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        
        candidate_pts = _unflatten_params(final_result.x)
        candidate_score = _score_points(candidate_pts)
        if candidate_score > best_score:
            best_score = candidate_score
            best_pts = candidate_pts.copy()
    except Exception:
        pass
    
    return best_pts
# EVOLVE-BLOCK-END
