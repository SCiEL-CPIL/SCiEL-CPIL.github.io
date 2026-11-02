"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

DEADLINE = 140.0
START_TIME = time.time()
SAFETY_MARGIN = 10.0


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0


def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area


def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
    """Generate points of a regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = center[0] + radius * np.cos(angles)
    ys = center[1] + radius * np.sin(angles)
    return np.column_stack([xs, ys])


def _neg_score(points_flat):
    """Negative score for minimization."""
    points = points_flat.reshape(-1, 2)
    return -_score(points)


def _optimize_from_start(start_points, max_iter=200):
    """Optimize from a starting configuration."""
    result = minimize(
        _neg_score,
        start_points.flatten(),
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    if result.success or result.fun < 0:
        return result.x.reshape(-1, 2)
    return start_points


def _make_initial_configurations():
    """Create several diverse initial configurations."""
    configs = []
    
    # Config 1: Regular 13-gon with center point
    # 12 on hull + 1 center = 13 points
    hull_pts = _regular_polygon(12, radius=1.0)
    center = np.array([[0.0, 0.0]])
    configs.append(np.vstack([hull_pts, center]))
    
    # Config 2: Regular 13-gon (all on hull)
    configs.append(_regular_polygon(13, radius=1.0))
    
    # Config 3: Regular 11-gon + 2 interior points
    hull_pts = _regular_polygon(11, radius=1.0)
    interior1 = np.array([[0.3, 0.0]])
    interior2 = np.array([[-0.15, 0.26]])
    configs.append(np.vstack([hull_pts, interior1, interior2]))
    
    # Config 4: Regular 10-gon + 3 interior points
    hull_pts = _regular_polygon(10, radius=1.0)
    interior = _regular_polygon(3, radius=0.3)
    configs.append(np.vstack([hull_pts, interior]))
    
    # Config 5: Regular 9-gon + 4 interior points
    hull_pts = _regular_polygon(9, radius=1.0)
    interior = _regular_polygon(4, radius=0.35)
    configs.append(np.vstack([hull_pts, interior]))
    
    # Config 6: Regular 13-gon with slight perturbation
    np.random.seed(42)
    base = _regular_polygon(13, radius=1.0)
    perturbed = base + np.random.normal(0, 0.05, base.shape)
    configs.append(perturbed)
    
    # Config 7: Two concentric rings (7 outer + 6 inner)
    outer = _regular_polygon(7, radius=1.0)
    inner = _regular_polygon(6, radius=0.5)
    configs.append(np.vstack([outer, inner]))
    
    # Config 8: 8 outer + 5 inner
    outer = _regular_polygon(8, radius=1.0)
    inner = _regular_polygon(5, radius=0.45)
    configs.append(np.vstack([outer, inner]))
    
    return configs


def construct():
    """Main entry point: find 13 points maximizing min triangle area / hull area."""
    deadline = START_TIME + DEADLINE
    
    configs = _make_initial_configurations()
    
    best_points = None
    best_score = -float('inf')
    
    for i, config in enumerate(configs):
        if time.time() > deadline - SAFETY_MARGIN:
            break
        
        # Evaluate initial config
        s = _score(config)
        if s > best_score:
            best_score = s
            best_points = config.copy()
        
        # Try to optimize
        if time.time() > deadline - SAFETY_MARGIN:
            break
        
        try:
            optimized = _optimize_from_start(config, max_iter=150)
            opt_score = _score(optimized)
            if opt_score > best_score:
                best_score = opt_score
                best_points = optimized.copy()
        except Exception:
            pass
    
    # If we have time, do a few more random restarts
    if time.time() < deadline - SAFETY_MARGIN:
        np.random.seed(123)
        for _ in range(5):
            if time.time() > deadline - SAFETY_MARGIN:
                break
            # Random convex-ish configuration
            n_outer = np.random.randint(6, 12)
            n_inner = N - n_outer
            outer = _regular_polygon(n_outer, radius=np.random.uniform(0.8, 1.2))
            if n_inner > 0:
                inner = _regular_polygon(n_inner, radius=np.random.uniform(0.2, 0.5))
                config = np.vstack([outer, inner])
            else:
                config = outer
            config = config + np.random.normal(0, 0.1, config.shape)
            
            s = _score(config)
            if s > best_score:
                best_score = s
                best_points = config.copy()
            
            if time.time() > deadline - SAFETY_MARGIN:
                break
            
            try:
                optimized = _optimize_from_start(config, max_iter=100)
                opt_score = _score(optimized)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = optimized.copy()
            except Exception:
                pass
    
    if best_points is None:
        best_points = _regular_polygon(13, radius=1.0)
    
    return best_points
# EVOLVE-BLOCK-END
