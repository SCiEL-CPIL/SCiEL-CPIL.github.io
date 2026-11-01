"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the Fibonacci lattice."""
    points = np.zeros((n, 3))
    phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle
    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        points[i, 0] = np.cos(theta) * radius
        points[i, 1] = y
        points[i, 2] = np.sin(theta) * radius
    return points

def _compute_worst_facet(points):
    """
    Compute the convex hull of the points and find the facet with the largest
    circumradius (angular distance from pole to vertex).
    Returns:
        max_radius: The maximum circumradius in radians.
        worst_vertices: Indices of the vertices of the worst facet.
        pole: The pole of the worst facet.
    """
    points = np.asarray(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi, [], np.array([1.0, 0.0, 0.0])
    
    max_radius = 0.0
    worst_verts = []
    worst_pole = np.array([1.0, 0.0, 0.0])
    
    for simplex in hull.simplices:
        verts = points[simplex]
        # Compute pole of the facet
        # For a triangle (simplex in 3D is always a triangle for ConvexHull in 3D? 
        # Yes, scipy ConvexHull in 3D returns triangular facets.)
        if len(verts) == 3:
            e1 = verts[1] - verts[0]
            e2 = verts[2] - verts[0]
            pole = np.cross(e1, e2)
            pole_norm = np.linalg.norm(pole)
            if pole_norm < 1e-12:
                continue
            pole = pole / pole_norm
            
            # Ensure pole is outward pointing (away from origin, which is inside hull)
            # The centroid of the facet is (v0+v1+v2)/3
            # Dot(pole, centroid) should be positive? 
            # Actually, for points on unit sphere, the pole is the normal.
            # The "outward" normal points away from the origin.
            # Since the origin is inside the hull, the normal pointing to the 
            # "outside" of the polytope is the one that has positive dot product 
            # with the centroid? No, the centroid is on the facet.
            # The vector from origin to centroid is roughly the pole direction.
            centroid = np.mean(verts, axis=0)
            if np.dot(pole, centroid) < 0:
                pole = -pole
                
            # Angular distance from pole to vertex 0
            cos_theta = np.clip(np.dot(verts[0], pole), -1.0, 1.0)
            theta = np.arccos(cos_theta)
            
            if theta > max_radius:
                max_radius = theta
                worst_verts = simplex
                worst_pole = pole
    
    return max_radius, worst_verts, worst_pole

def construct():
    # Initialize with Fibonacci sphere
    points = _fibonacci_sphere(N)
    
    # Local search parameters
    n_iterations = 3000
    step_size = 0.05
    decay_rate = 0.999
    rng = np.random.default_rng(42)
    
    best_radius = np.pi
    
    for i in range(n_iterations):
        # Compute worst facet
        current_radius, worst_verts, pole = _compute_worst_facet(points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            # Optional: store best points
            # best_points = points.copy()
            
        # If no valid facet found or radius is very small, break
        if len(worst_verts) == 0 or current_radius < 1e-6:
            break
        
        # Move vertices of the worst facet towards the pole
        # We update a copy to ensure consistency
        new_points = points.copy()
        
        for idx in worst_verts:
            v = points[idx]
            # Direction from v to pole projected onto tangent plane at v
            # proj = pole - dot(pole, v) * v
            proj = pole - np.dot(pole, v) * v
            proj_norm = np.linalg.norm(proj)
            if proj_norm < 1e-12:
                continue
            direction = proj / proj_norm
            
            # Move
            new_v = v + step_size * direction
            # Normalize
            new_v = new_v / np.linalg.norm(new_v)
            new_points[idx] = new_v
        
        points = new_points
        
        # Decay step size
        step_size *= decay_rate
        
        # Occasionally add small random perturbations to escape local minima
        if i % 100 == 50 and step_size > 0.001:
            perturb = rng.standard_normal((N, 3)) * 0.01
            points = points + perturb
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
            
    # Final cleanup: ensure unit norm
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    return points
# EVOLVE-BLOCK-END
