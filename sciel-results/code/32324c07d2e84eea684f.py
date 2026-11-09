"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time


def _normalize(v):
    """Normalize vectors to unit length."""
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return v / norms


def _spherical_circumcenter(points_2d):
    """Compute the spherical circumcenter of a spherical triangle.
    
    For three points on the unit sphere, the circumcenter is the normalized
    cross product of (b-a) and (c-a), which gives the normal to the plane
    containing the three points. This is the center of the unique circle
    passing through all three points on the sphere.
    """
    a, b, c = points_2d
    # The circumcenter direction is the normalized cross product
    normal = np.cross(b - a, c - a)
    norm_val = np.linalg.norm(normal)
    if norm_val < 1e-12:
        # Degenerate case: return centroid
        center = a + b + c
        center_norm = np.linalg.norm(center)
        if center_norm < 1e-12:
            return a
        return center / center_norm
    return normal / norm_val


def _facet_circumradius(points, simplex):
    """Compute the circumradius of a facet (in radians) on the unit sphere."""
    if len(simplex) != 3:
        return 0.0
    facet_pts = points[simplex]
    center = _spherical_circumcenter(facet_pts)
    # The circumradius is the angular distance from center to any vertex
    cos_dists = np.clip(np.dot(facet_pts, center), -1.0, 1.0)
    # All should be equal; take max for robustness
    return np.arccos(cos_dists).max()


def _covering_radius_degrees(points):
    """Compute covering radius in degrees using convex hull facets."""
    try:
        hull = ConvexHull(points)
        max_radius = 0.0
        for simplex in hull.simplices:
            if len(simplex) == 3:
                cr = _facet_circumradius(points, simplex)
                max_radius = max(max_radius, cr)
        return np.degrees(max_radius)
    except Exception:
        return 90.0


def _objective(points):
    """Objective: covering radius in degrees."""
    return _covering_radius_degrees(points)


def _equalize_circumradii_step(points, lr=0.02):
    """Move vertices of high-circumradius facets toward their circumcenters.
    
    This is a dual-polytope approach: instead of moving points based on
    nearest neighbors, we directly act on the facets (dual cells) to
    equalize their circumradii, which directly controls the covering radius.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return points.copy()
    
    new_points = points.copy()
    
    # Compute circumradius for each facet
    facet_radii = []
    for simplex in hull.simplices:
        if len(simplex) == 3:
            cr = _facet_circumradius(points, simplex)
            facet_radii.append((cr, simplex))
    
    if not facet_radii:
        return points.copy()
    
    # Find the maximum circumradius
    max_cr = max(cr for cr, _ in facet_radii)
    
    # For each facet with above-average circumradius, move its vertices
    # toward the facet's circumcenter
    for cr, simplex in facet_radii:
        if cr > 0.7 * max_cr:  # Only act on large facets
            facet_pts = points[simplex]
            center = _spherical_circumcenter(facet_pts)
            
            # Move each vertex toward the circumcenter
            for idx in simplex:
                p = points[idx]
                # Slerp-like step: move p toward center by a fraction
                # of the angular distance
                cos_angle = np.clip(np.dot(p, center), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                if angle > 1e-10:
                    # Move by lr * angle toward center
                    step_size = lr * angle
                    if step_size < angle:
                        # Slerp
                        sin_angle = np.sin(angle)
                        if sin_angle < 1e-12:
                            new_p = center
                        else:
                            new_p = (np.sin((1 - step_size/angle) * angle) * p + 
                                     np.sin(step_size * angle) * center) / sin_angle
                        new_points[idx] = new_p
    
    return _normalize(new_points)


def _voronoi_optimize(points, max_iters=2000, time_limit=140.0):
    """Iteratively optimize using dual-polytope circumradius equalization."""
    start_time = time.time()
    best_points = points.copy()
    best_score = _objective(points)
    
    lr = 0.03
    stagnant_count = 0
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        new_points = _equalize_circumradii_step(points, lr)
        new_score = _objective(new_points)
        
        if new_score < best_score - 1e-10:
            best_score = new_score
            best_points = new_points.copy()
            points = new_points
            stagnant_count = 0
            lr = max(0.005, lr * 0.98)
        else:
            stagnant_count += 1
            if stagnant_count > 50:
                # Restart with perturbation
                rng = np.random.RandomState(it)
                perturbed = points + rng.randn(N, 3) * 0.01
                points = _normalize(perturbed)
                stagnant_count = 0
                lr = 0.03
            else:
                lr = max(0.005, lr * 0.95)
    
    return best_points, best_score


def _initial_config():
    """Generate a good initial configuration using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)


def construct():
    """Construct 50 points on the unit sphere minimizing covering radius."""
    start_time = time.time()
    
    # Generate initial configuration
    points = _initial_config()
    
    # Run dual-polytope optimization
    best_points, best_score = _voronoi_optimize(points, max_iters=3000, time_limit=145.0)
    
    # Ensure we return valid points
    if time.time() - start_time > 148.0:
        return _normalize(best_points)
    
    return _normalize(best_points)
# EVOLVE-BLOCK-END
