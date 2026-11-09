import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_covering_metrics(points):
    """
    Computes the covering radius and identifies the worst facet.
    Returns: (radius, worst_facet_indices, circumcenter)
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback if hull fails (e.g., coplanar points), return large radius
        return np.pi, None, None
    
    max_radius = 0.0
    worst_facet = None
    worst_center = None
    
    # hull.simplices gives indices of vertices for each facet
    for simplex in hull.simplices:
        v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        
        # Compute circumcenter of the triangle on the sphere
        # The circumcenter is in the plane of the triangle.
        # Normal to the plane: n = cross(v1-v2, v1-v3)
        # Circumcenter c = n * t
        # We need c such that |c - v1| = |c - v2| = |c - v3|
        # Actually, for points on the unit sphere, the circumcenter of the triangle
        # formed by them is the intersection of the plane of the triangle with the line
        # through the origin perpendicular to the plane? No.
        # The circumcenter of a spherical triangle is the point on the sphere equidistant from the vertices.
        # It lies on the normal to the plane of the triangle passing through the origin?
        # Let's derive: c = k * n, where n = cross(v1-v2, v1-v3).
        # c is on the unit sphere.
        # Distance squared from c to v1 is |c|^2 + |v1|^2 - 2 c.dot(v1) = 2 - 2 c.dot(v1).
        # We want to minimize this? No, the circumcenter is unique.
        # The vector n is perpendicular to the plane. The circumcenter lies on the line spanned by n.
        # So c = ± n / |n|.
        # Which sign? The one that is closer to the vertices?
        # Actually, the circumcenter of the spherical triangle is the normalized cross product of (v2-v1) and (v3-v1)?
        # Let's use the formula for circumcenter of a triangle in 3D.
        # Let a = v1, b = v2, c = v3.
        # The circumcenter O satisfies |O-a|^2 = |O-b|^2 = |O-c|^2.
        # 2 O.(b-a) = |b|^2 - |a|^2 = 0 (since |a|=|b|=1)
        # 2 O.(c-a) = 0
        # So O is perpendicular to b-a and c-a. Thus O is parallel to (b-a) x (c-a).
        # n = np.cross(v2 - v1, v3 - v1)
        # c_circ = n / np.linalg.norm(n)
        # We need to ensure this is the correct pole (the one "inside" the spherical triangle if it's small, or just the one that gives the correct radius).
        # The radius is the angle between c_circ and v1.
        
        n = np.cross(v2 - v1, v3 - v1)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            continue
        c_circ = n / norm_n
        
        # Check if c_circ is on the same side of the plane as the origin?
        # The origin is (0,0,0). The plane equation is n.(x - v1) = 0 => n.x = n.v1.
        # n.0 = 0. If n.v1 > 0, origin is on one side.
        # The circumcenter should be the one such that the triangle vertices are "around" it.
        # For a convex hull on the sphere, the facets are visible from the origin.
        # The normal n points outwards?
        # Let's check the dot product of c_circ with the vertices.
        # If the triangle is small, the circumcenter is close to the centroid.
        # Let's just compute the radius for both c_circ and -c_circ and pick the one with smaller radius?
        # No, the covering radius is the max distance to the *nearest* point.
        # The Voronoi vertex is the circumcenter.
        # The distance from the circumcenter to the vertices is the circumradius.
        # We want to minimize the max circumradius.
        # So we calculate the angle between c_circ and v1.
        
        # Ensure c_circ is the correct one. 
        # The origin is inside the tetrahedron formed by origin and the facet?
        # For a convex hull of points on the sphere, the origin is inside the hull.
        # The facet normal should point away from the origin.
        # n = cross(v2-v1, v3-v1).
        # If n.dot(v1) < 0, then n points towards the origin?
        # Let's check: v1 is on the sphere. n is perpendicular to the plane.
        # If n.dot(v1) > 0, the normal points away from the origin (since origin is 0, and v1 is positive dot).
        # Wait, the plane passes through v1. The vector from origin to v1 is v1.
        # If n.dot(v1) > 0, the normal has a component in the direction of v1.
        # The circumcenter c_circ = n/|n|.
        # The distance is arccos(c_circ.dot(v1)).
        # If we pick the wrong sign, the distance will be pi - arccos(...).
        # We want the smaller distance, which corresponds to the circumcenter of the spherical triangle.
        # For a convex hull, the "cap" is the small cap. So we want the circumcenter that is close to the triangle.
        # This implies c_circ should have a positive dot product with the centroid of the triangle?
        # Centroid g = (v1+v2+v3)/3.
        # If c_circ.dot(g) < 0, flip c_circ.
        
        centroid = (v1 + v2 + v3) / 3.0
        if np.dot(c_circ, centroid) < 0:
            c_circ = -c_circ
            
        # Calculate radius (angle)
        cos_theta = np.clip(np.dot(c_circ, v1), -1.0, 1.0)
        radius = np.arccos(cos_theta)
        
        if radius > max_radius:
            max_radius = radius
            worst_facet = simplex
            worst_center = c_circ
            
    return max_radius, worst_facet, worst_center

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    # Initial configuration
    points = _fibonacci_sphere(N)
    best_points = points.copy()
    best_radius, _, _ = _compute_covering_metrics(best_points)
    
    # Optimization parameters
    step_size = 0.01
    decay = 0.999
    
    while time.time() < deadline:
        current_radius, worst_facet, worst_center = _compute_covering_metrics(points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        
        if worst_facet is None:
            break
            
        v1_idx, v2_idx, v3_idx = worst_facet
        v1, v2, v3 = points[v1_idx], points[v2_idx], points[v3_idx]
        
        # Move points towards the circumcenter
        # We want to reduce the circumradius.
        # Moving towards the center reduces the distance to the center.
        # But we must stay on the sphere.
        # p_new = normalize(p + alpha * (c - p))
        
        alpha = step_size
        p1_new = v1 + alpha * (worst_center - v1)
        p2_new = v2 + alpha * (worst_center - v2)
        p3_new = v3 + alpha * (worst_center - v3)
        
        # Normalize
        n1 = np.linalg.norm(p1_new)
        n2 = np.linalg.norm(p2_new)
        n3 = np.linalg.norm(p3_new)
        
        if n1 > 1e-10 and n2 > 1e-10 and n3 > 1e-10:
            points[v1_idx] = p1_new / n1
            points[v2_idx] = p2_new / n2
            points[v3_idx] = p3_new / n3
        
        # Decay step size to ensure convergence
        step_size *= decay
        
        # If step size is too small, break
        if step_size < 1e-6:
            break
            
    return best_points
