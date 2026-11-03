# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return points / norms

def get_circumcenter_and_radius(facet_points):
    """
    Compute the circumcenter on the unit sphere and the angular radius
    for a triangle defined by 3 points on the sphere.
    """
    a, b, c = facet_points
    # Vectors from a to b and a to c
    # The circumcenter is the point equidistant from a, b, c.
    # In 3D, the circumcenter of the triangle formed by a,b,c on the sphere
    # lies on the line passing through the origin and the centroid of the 
    # "spherical" triangle? No.
    # It is the intersection of the sphere with the plane perpendicular to the 
    # great circle arcs?
    # Actually, the circumcenter $o$ satisfies $o \cdot a = o \cdot b = o \cdot c = \cos(R)$.
    # So $o$ is in the null space of the matrix with rows (a-b, b-c, c-a)?
    # Let's solve the linear system:
    # (a-b) . o = 0
    # (b-c) . o = 0
    # o . o = 1
    # The first two equations mean o is perpendicular to the plane containing a,b,c?
    # No, the plane containing a,b,c does not pass through the origin generally.
    # The condition is that the projection of o onto the plane of the triangle 
    # is the circumcenter of the Euclidean triangle? No.
    
    # Correct approach:
    # o is a linear combination of a, b, c: o = x*a + y*b + z*c
    # o . a = o . b = o . c
    # This implies:
    # x(a.a) + y(b.a) + z(c.a) = x(a.b) + y(b.b) + z(c.b)
    # x(a.b) + y(b.b) + z(c.b) = x(a.c) + y(b.c) + z(c.c)
    # Let G be the Gram matrix of a,b,c.
    # G * [x,y,z]^T = [1,1,1]^T is not quite right because o is not unit yet.
    # We want o such that o.a = o.b = o.c.
    # This means (a-b).o = 0 and (b-c).o = 0.
    # So o is orthogonal to the vectors a-b and b-c.
    # Thus o is parallel to the cross product (a-b) x (b-c).
    
    v1 = a - b
    v2 = b - c
    normal = np.cross(v1, v2)
    if np.linalg.norm(normal) < 1e-10:
        # Degenerate case, points are nearly collinear on a great circle
        # Return a point on the great circle, e.g., midpoint of the longest arc
        # For now, return 0 radius or handle carefully
        return np.zeros(3), 0.0
        
    o = normal / np.linalg.norm(normal)
    
    # Check which side of the hemisphere the center is on.
    # The circumcenter should be "inside" the triangle.
    # We check the dot product with the centroid of the triangle (Euclidean)
    centroid = (a + b + c) / 3.0
    if np.dot(o, centroid) < 0:
        o = -o
        
    # The radius is the angular distance from o to a
    cos_r = np.clip(np.dot(o, a), -1.0, 1.0)
    r = np.arccos(cos_r)
    
    return o, r

def compute_covering_radius(points):
    """
    Compute the covering radius and the indices of the points in the 
    worst facet.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback if hull fails (e.g., all points in a hemisphere)
        # Use max distance to a random sample
        rng = np.random.RandomState(42)
        samples = rng.randn(1000, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        d = np.max(np.min(np.arccos(np.clip(np.dot(samples, points.T), -1, 1)), axis=1))
        return d, None

    max_radius = 0.0
    worst_facet = None
    
    for simplex in hull.simplices:
        facet_pts = points[simplex]
        center, radius = get_circumcenter_and_radius(facet_pts)
        if radius > max_radius:
            max_radius = radius
            worst_facet = simplex

    return max_radius, worst_facet

def construct():
    # Initial configuration
    points = fibonacci_sphere(N)
    
    best_points = points.copy()
    best_radius, _ = compute_covering_radius(best_points)
    
    # Optimization loop
    # Use a simple iterative improvement:
    # 1. Find the worst facet.
    # 2. Move the points in that facet towards the circumcenter.
    # 3. Occasionally, perform a random perturbation to escape local minima.
    
    steps = 2000
    learning_rate = 0.01
    
    for i in range(steps):
        radius, worst_facet = compute_covering_radius(points)
        
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
            
        if worst_facet is None:
            # If we can't compute the hull, break or restart
            break
            
        # Get the circumcenter of the worst facet
        facet_pts = points[worst_facet]
        center, _ = get_circumcenter_and_radius(facet_pts)
        
        # Move the points in the worst facet towards the center
        # This reduces the circumradius of that facet
        for idx in worst_facet:
            points[idx] = points[idx] + learning_rate * center
            points[idx] /= np.linalg.norm(points[idx])
            
        # Occasionally, add noise to escape local minima
        if i % 100 == 99:
            noise = np.random.randn(N, 3) * 0.05
            points += noise
            points = normalize(points)
            
    return best_points
# EVOLVE-BLOCK-END
