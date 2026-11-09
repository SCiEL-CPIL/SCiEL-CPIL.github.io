import numpy as np
from scipy.spatial import ConvexHull
import time

N = 14

# EVOLVE-BLOCK-START

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by triples of points."""
    n = points.shape[0]
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = points[i]
                bx, by = points[j]
                cx, cy = points[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area


def _regular_polygon(n):
    """Generate vertices of a regular n-gon."""
    angles = 2 * np.pi * np.arange(n) / n
    return np.column_stack((np.cos(angles), np.sin(angles)))


def _jitter_polygon(n, rng, max_jitter=0.1):
    """Generate a jittered regular n-gon."""
    base = _regular_polygon(n)
    angles = 2 * np.pi * np.arange(n) / n + rng.uniform(-max_jitter, max_jitter, n)
    angles = np.sort(angles)
    # Normalize to start from 0
    angles -= angles[0]
    return np.column_stack((np.cos(angles), np.sin(angles)))


def _powell_local_optimize(points, max_iter=500, tol=1e-8):
    """Local optimization using Powell's method to maximize the score."""
    from scipy.optimize import minimize
    
    def negative_score(x):
        pts = x.reshape(-1, 2)
        return -_score(pts)
    
    x0 = points.flatten()
    result = minimize(negative_score, x0, method='Powell', options={'maxiter': max_iter, 'xtol': tol, 'ftol': tol})
    
    if result.success:
        return result.x.reshape(-1, 2)
    else:
        # Check if the result is still better than initial
        if -result.fun > _score(points):
            return result.x.reshape(-1, 2)
        return points


def _refine_with_perturbation(points, n_perturbations=20, max_iter=200):
    """Try multiple small perturbations and keep the best."""
    best_points = points
    best_score = _score(points)
    rng = np.random.default_rng(42)
    
    for _ in range(n_perturbations):
        # Small random perturbation
        perturbed = points + rng.normal(0, 0.05, points.shape)
        # Try local optimization from this perturbed state
        optimized = _powell_local_optimize(perturbed, max_iter=max_iter)
        s = _score(optimized)
        if s > best_score:
            best_score = s
            best_points = optimized
    
    return best_points


def construct():
    deadline = time.time() + 140  # 10s safety margin
    
    # Start with regular polygon
    best_points = _regular_polygon(N)
    best_score = _score(best_points)
    
    # Try local optimization from regular polygon
    if time.time() < deadline:
        optimized = _powell_local_optimize(best_points, max_iter=1000)
        s = _score(optimized)
        if s > best_score:
            best_score = s
            best_points = optimized
    
    # Try jittered polygons
    rng = np.random.default_rng(N)
    n_trials = 0
    while time.time() < deadline:
        n_trials += 1
        jittered = _jitter_polygon(N, rng, max_jitter=0.15)
        s = _score(jittered)
        if s > best_score:
            best_score = s
            best_points = jittered
        
        # Try local optimization from this jittered state
        if time.time() < deadline:
            optimized = _powell_local_optimize(jittered, max_iter=300)
            s = _score(optimized)
            if s > best_score:
                best_score = s
                best_points = optimized
        
        # Limit trials to avoid infinite loop
        if n_trials >= 30:
            break
    
    # Final refinement with perturbation
    if time.time() < deadline - 10:
        refined = _refine_with_perturbation(best_points, n_perturbations=10, max_iter=100)
        s = _score(refined)
        if s > best_score:
            best_score = s
            best_points = refined
    
    return best_points

# EVOLVE-BLOCK-END
