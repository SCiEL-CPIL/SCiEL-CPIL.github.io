import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

N = 130

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin from 150s
    
    # 1. Initial Configuration: Fibonacci Spiral
    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    points = fibonacci_spiral(N)
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    def calc_covering_radius(pts):
        """
        Calculate the covering radius by finding the maximum circumradius
        of the facets of the convex hull.
        """
        try:
            hull = ConvexHull(pts)
            max_rad = 0.0
            max_facet_idx = -1
            for simplex in hull.simplices:
                # simplex is a set of indices, usually 3 for 3D
                tri = pts[simplex]
                # Calculate circumradius of the triangle
                # Formula: R = (a*b*c) / (4*Area)
                # a, b, c are side lengths
                a = np.linalg.norm(tri[0] - tri[1])
                b = np.linalg.norm(tri[1] - tri[2])
                c = np.linalg.norm(tri[2] - tri[0])
                
                # Area using cross product
                cross = np.cross(tri[0] - tri[1], tri[0] - tri[2])
                area = 0.5 * np.linalg.norm(cross)
                
                if area > 1e-12:
                    rad = (a * b * c) / (4.0 * area)
                else:
                    rad = 0.0
                    
                if rad > max_rad:
                    max_rad = rad
                    max_facet_idx = len(hull.simplices) - 1 # Need to track index properly
                    
            # We need the actual indices of the worst facet
            # Let's re-do to return indices
            max_rad = 0.0
            worst_simplex = None
            for simplex in hull.simplices:
                tri = pts[simplex]
                a = np.linalg.norm(tri[0] - tri[1])
                b = np.linalg.norm(tri[1] - tri[2])
                c = np.linalg.norm(tri[2] - tri[0])
                cross = np.cross(tri[0] - tri[1], tri[0] - tri[2])
                area = 0.5 * np.linalg.norm(cross)
                if area > 1e-12:
                    rad = (a * b * c) / (4.0 * area)
                else:
                    rad = 0.0
                if rad > max_rad:
                    max_rad = rad
                    worst_simplex = simplex
                    
            return max_rad, worst_simplex, hull
            
        except Exception:
            # Fallback: sample points if hull fails (degenerate)
            samples = np.random.randn(10000, 3)
            samples /= np.linalg.norm(samples, axis=1, keepdims=True)
            dists = cdist(samples, pts)
            min_dists = np.min(dists, axis=1)
            max_dist = np.max(min_dists)
            # Approximate angle from chord length: 2*arcsin(d/2)
            rad = 2.0 * np.arcsin(max_dist / 2.0)
            return rad, None, None

    # Initial radius
    current_radius, worst_simplex, _ = calc_covering_radius(points)
    best_points = points.copy()
    best_radius = current_radius
    
    step_size = 0.05
    stuck_counter = 0
    
    while time.time() - start_time < time_limit:
        # Check time
        if time.time() - start_time > time_limit - 5.0:
            break
            
        # Get the worst facet
        current_radius, worst_simplex, hull = calc_covering_radius(points)
        
        if worst_simplex is None:
            # Degenerate case, try random perturbation
            noise = np.random.randn(N, 3) * 0.01
            points = points + noise
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            continue
            
        # Move vertices of the worst facet towards its circumcenter
        tri = points[worst_simplex]
        
        # Calculate circumcenter of the triangle on the sphere
        # The circumcenter is the intersection of the sphere and the plane passing through the 3 points?
        # No, the circumcenter of the spherical triangle is the point on the sphere equidistant from the 3 vertices.
        # It is the normalized vector sum of the vertices? No, that's the centroid.
        # The circumcenter of a spherical triangle is the pole of the great circle circumscribing it.
        # It is proportional to the cross product of the edges?
        # Let's use the property: The circumcenter is the point $c$ such that $c \cdot v_1 = c \cdot v_2 = c \cdot v_3$.
        # This implies $c$ is orthogonal to $v_1 - v_2$ and $v_2 - v_3$.
        # So $c$ is parallel to $(v_1 - v_2) \times (v_2 - v_3)$.
        
        v1, v2, v3 = tri
        c_vec = np.cross(v1 - v2, v2 - v3)
        c_norm = np.linalg.norm(c_vec)
        
        if c_norm < 1e-9:
            # Degenerate triangle (collinear or very small)
            # Move towards centroid
            c_vec = (v1 + v2 + v3) / 3.0
            c_norm = np.linalg.norm(c_vec)
            
        if c_norm < 1e-9:
            break # Should not happen
            
        c_vec /= c_norm
        
        # Check if c_vec is on the "same side" as the normal
        # The center of the cap should be in the direction of the normal of the facet
        # The normal of the facet is c_vec. The cap center is c_vec (or -c_vec).
        # We want to move points TOWARDS the cap center to reduce the radius.
        # So we move v_i towards c_vec.
        
        new_points = points.copy()
        
        for idx in worst_simplex:
            v = new_points[idx]
            # Move towards c_vec
            # Step size should be small
            new_v = v + step_size * (c_vec - v)
            new_v /= np.linalg.norm(new_v)
            new_points[idx] = new_v
            
        # Evaluate new radius
        new_radius, _, _ = calc_covering_radius(new_points)
        
        if new_radius < current_radius:
            points = new_points
            current_radius = new_radius
            stuck_counter = 0
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
            # Slightly increase step size if we are improving fast?
            # Or keep constant.
        else:
            stuck_counter += 1
            # Reduce step size if stuck
            if stuck_counter > 50:
                step_size *= 0.9
                stuck_counter = 0
                
            # If step size becomes too small, try a random perturbation
            if step_size < 1e-4:
                # Random restart or perturbation
                noise = np.random.randn(N, 3) * 0.05
                points = points + noise
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                step_size = 0.05
                stuck_counter = 0

    return best_points
