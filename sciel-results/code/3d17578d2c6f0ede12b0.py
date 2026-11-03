"""Improved program for sphere covering with 50 points."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, minimize_scalar
import math


def _covering_radius(points):
    """Compute the covering radius in degrees for points on the unit sphere."""
    # Compute convex hull
    try:
        hull = ConvexHull(points)
    except Exception:
        return 180.0  # invalid configuration
    
    # For each facet, compute the circumradius
    # A facet is defined by 3 points (triangular facets on sphere)
    max_circumradius = 0.0
    facets = hull.simplices
    
    for simplex in facets:
        # Get the three points forming this facet
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]
        
        # Compute the normal to the facet
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-15:
            continue
        normal = normal / norm
        
        # The farthest point from the facet center on the sphere
        # is along the normal direction (outward from origin)
        # The angular distance from the facet center to the sphere surface
        # is the angle between the normal and the facet plane
        
        # Compute the distance from origin to the facet plane
        d = np.dot(p1, normal)
        
        # If d <= 0, the origin is on or below the facet plane, 
        # meaning the facet is not visible from outside, or we have an issue
        if d <= 0:
            # This shouldn't happen for a proper convex hull containing origin
            # but handle it gracefully
            d = abs(d)
        
        # The circumradius is the angle whose cosine is d (since sphere radius = 1)
        # cos(theta) = d / 1 = d
        # But we need to be careful: the circumradius is the maximum angular distance
        # from any point on the facet to its vertices, which equals the angle between
        # the normal and the line from origin to the farthest vertex
        
        # Actually, for a spherical cap defined by a facet, the covering radius contribution
        # is the angular distance from the facet center (along normal) to the farthest vertex
        # This is arcsin of the ratio of the circumradius of the triangle to the distance
        
        # Simpler approach: the maximum angular distance on this facet to any vertex
        # is the maximum of the angular distances from the facet center to each vertex
        # The facet center is at normal * d (projected onto sphere: normal)
        # Wait, the point on the sphere closest to the facet plane along the normal is 'normal'
        # (since it's a unit sphere). The angular distance from 'normal' to a vertex p is
        # arccos(dot(normal, p))
        
        for p in [p1, p2, p3]:
            cos_angle = np.dot(normal, p)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            if angle > max_circumradius:
                max_circumradius = angle
    
    return np.degrees(max_circumradius)


def _soft_covering_radius(points, temperature=0.1):
    """Soft surrogate for covering radius using log-sum-exp."""
    # Compute all pairwise angular distances
    dot_products = points @ points.T
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angles = np.arccos(dot_products)
    
    # For each point, find the minimum distance to any other point
    # But we want the covering radius, which is about the gaps between points
    
    # Alternative: use the convex hull approach but with a smooth approximation
    # For now, fall back to exact computation
    return _covering_radius(points)


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    best_points = points.copy()
    best_radius = _covering_radius(best_points)
    
    # Local optimization using gradient-free methods
    def objective(x):
        pts = x.reshape(N, 3)
        # Normalize to sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        pts = pts / norms
        return _covering_radius(pts)
    
    # Try Nelder-Mead optimization
    try:
        result = minimize(
            objective,
            points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
        )
        if result.fun < best_radius:
            best_points = result.x.reshape(N, 3)
            best_radius = result.fun
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        # Return best found
        norms = np.linalg.norm(best_points, axis=1, keepdims=True)
        best_points = best_points / np.maximum(norms, 1e-10)
        return best_points
    
    # Try a second round with different initial perturbations
    np.random.seed(42)
    for restart in range(3):
        if time.time() > deadline:
            break
        
        # Perturb the best points
        noise = np.random.randn(N, 3) * 0.1
        perturbed = best_points + noise
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        perturbed = perturbed / np.maximum(norms, 1e-10)
        
        try:
            result = minimize(
                objective,
                perturbed.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if result.fun < best_radius:
                best_points = result.x.reshape(N, 3)
                best_radius = result.fun
        except Exception:
            pass
    
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / np.maximum(norms, 1e-10)
    
    return best_points
# EVOLVE-BLOCK-END
