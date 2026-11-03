"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import sys


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros(3)
    return v / n


def _compute_covering_radius(points):
    """
    Compute the covering radius in degrees.
    The covering radius is the maximum over all points on the sphere of the
    angular distance to the nearest configuration point.
    
    This equals the maximum circumradius of the spherical triangles (facets)
    of the convex hull, where each facet is a spherical triangle formed by
    three points on the unit sphere.
    
    We use the fact that for a set of points on the unit sphere, the covering
    radius is determined by the largest empty spherical cap, which corresponds
    to the largest circumcircle of the Delaunay triangulation on the sphere.
    
    For practical computation, we can use the convex hull in 3D. The facets
    of the convex hull correspond to the Delaunay triangles on the sphere.
    For each facet (triangle), we compute the circumradius of the spherical
    triangle.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        # If hull computation fails, fall back to a different method
        return _compute_covering_radius_fallback(points)
    
    max_radius = 0.0
    
    for simplex in hull.simplices:
        p0 = points[simplex[0]]
        p1 = points[simplex[1]]
        p2 = points[simplex[2]]
        
        # Compute the circumcenter of the spherical triangle
        # The circumcenter is the point equidistant from all three vertices
        # on the sphere. It's the intersection of the perpendicular bisectors.
        
        # For a spherical triangle, the circumradius can be computed as follows:
        # Find the point C on the sphere that is equidistant from p0, p1, p2.
        # This means C is perpendicular to (p0 - p1), (p1 - p2), etc. in some sense.
        
        # Actually, the circumcenter of a spherical triangle is the point C such that
        # the angular distances from C to p0, p1, p2 are all equal.
        
        # We can find C by solving for a unit vector c such that c·p0 = c·p1 = c·p2.
        # This means c is perpendicular to (p0 - p1) and (p0 - p2).
        # So c is parallel to (p0 - p1) × (p0 - p2).
        
        v1 = p0 - p1
        v2 = p0 - p2
        cross = np.cross(v1, v2)
        cross_norm = np.linalg.norm(cross)
        
        if cross_norm < 1e-12:
            # Degenerate triangle
            continue
        
        c = cross / cross_norm
        
        # The circumradius is the angular distance from c to any of p0, p1, p2
        # But we need to be careful: c could be on the "wrong" side
        # We need to check which hemisphere c is in
        
        # The circumcenter should be such that the triangle is contained in
        # the hemisphere centered at c. Let's check the dot products.
        d0 = np.dot(c, p0)
        d1 = np.dot(c, p1)
        d2 = np.dot(c, p2)
        
        # If any dot product is negative, the circumcenter is on the opposite side
        # and we should use -c
        if d0 < 0 or d1 < 0 or d2 < 0:
            c = -c
            d0 = -d0
            d1 = -d1
            d2 = -d2
        
        # The circumradius is arccos(d0) (should be same for d1, d2)
        # Clamp to valid range
        d0 = np.clip(d0, -1.0, 1.0)
        radius = np.arccos(d0)
        
        if radius > max_radius:
            max_radius = radius
    
    return np.degrees(max_radius)


def _compute_covering_radius_fallback(points):
    """
    Fallback method: sample points on the sphere and compute the maximum
    distance to the nearest configuration point.
    """
    # Use a dense grid of sample points
    # Create a grid of points on the sphere
    n_samples = 10000
    rng = np.random.RandomState(42)
    
    # Use Fibonacci sphere for deterministic sampling
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute distances from each sample to all points
    # Use dot product for efficiency
    dots = samples @ points.T  # shape (n_samples, N)
    max_dist = 0.0
    for i in range(n_samples):
        min_dot = np.max(dots[i])
        min_dot = np.clip(min_dot, -1.0, 1.0)
        dist = np.arccos(min_dot)
        if dist > max_dist:
            max_dist = dist
    
    return np.degrees(max_dist)


def _worst_facet_circumcenter_move(points, step_size=0.01):
    """
    Find the facet with the largest circumradius and move its vertices
    toward the circumcenter.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return points, 0.0
    
    max_radius = 0.0
    worst_simplex = None
    worst_circumcenter = None
    
    for simplex in hull.simplices:
        p0 = points[simplex[0]]
        p1 = points[simplex[1]]
        p2 = points[simplex[2]]
        
        v1 = p0 - p1
        v2 = p0 - p2
        cross = np.cross(v1, v2)
        cross_norm = np.linalg.norm(cross)
        
        if cross_norm < 1e-12:
            continue
        
        c = cross / cross_norm
        
        d0 = np.dot(c, p0)
        d1 = np.dot(c, p1)
        d2 = np.dot(c, p2)
        
        if d0 < 0 or d1 < 0 or d2 < 0:
            c = -c
            d0 = -d0
            d1 = -d1
            d2 = -d2
        
        d0 = np.clip(d0, -1.0, 1.0)
        radius = np.arccos(d0)
        
        if radius > max_radius:
            max_radius = radius
            worst_simplex = simplex
            worst_circumcenter = c
    
    if worst_simplex is None:
        return points, 0.0
    
    # Move the three vertices toward the circumcenter
    new_points = points.copy()
    for idx in worst_simplex:
        p = points[idx]
        # Move along the great circle toward the circumcenter
        # The direction of movement is the component of the circumcenter
        # tangent to the sphere at p
        tangent = worst_circumcenter - np.dot(worst_circumcenter, p) * p
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > 1e-12:
            tangent = tangent / tangent_norm
            # Update the point
            new_p = p + step_size * tangent
            new_points[idx] = _normalize(new_p)
    
    return new_points, max_radius


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Iterative optimization: worst-facet circumcenter projection
    step_size = 0.05
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        new_points, current_radius = _worst_facet_circumcenter_move(points, step_size)
        
        new_radius = _compute_covering_radius(new_points)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_points = new_points.copy()
            points = new_points
            # Decrease step size gradually
            if iteration % 100 == 0 and step_size > 0.001:
                step_size *= 0.95
        else:
            # Accept with some probability to escape local minima (simulated annealing)
            # Or just reject and try a smaller step
            step_size *= 0.9
            if step_size < 0.001:
                break
    
    # Final refinement with smaller steps
    step_size = 0.005
    for iteration in range(2000):
        if time.time() > deadline:
            break
        
        new_points, current_radius = _worst_facet_circumcenter_move(points, step_size)
        new_radius = _compute_covering_radius(new_points)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_points = new_points.copy()
            points = new_points
        else:
            if step_size < 0.0005:
                break
            step_size *= 0.9
    
    return best_points
# EVOLVE-BLOCK-END
