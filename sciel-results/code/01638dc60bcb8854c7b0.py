import numpy as np
import scipy.spatial
import time

N = 130

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Helper to compute covering radius and worst facet
    def get_covering_info(pts):
        # Normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        
        try:
            hull = scipy.spatial.ConvexHull(pts)
        except:
            return 1000.0, -1, None
            
        max_rad = 0.0
        worst_idx = -1
        worst_circumcenter = None
        
        for idx, simplex in enumerate(hull.simplices):
            v1, v2, v3 = pts[simplex]
            n = np.cross(v1 - v2, v1 - v3)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            n = n / n_norm
            
            # Ensure n points "outward" (away from origin)
            # The origin is inside the hull if the points are well-distributed.
            # The outward normal should have positive dot product with the vertices?
            # Actually, for a convex hull of points on the sphere, the origin is inside.
            # The normal n computed by cross(v1-v2, v1-v3) is perpendicular to the facet.
            # We need the one pointing away from the origin.
            # If n . v1 > 0, it points towards the vertex side.
            # The cap is on the side of the normal.
            # The circumcenter is n (if n.v1 > 0) or -n.
            # Let's check which one is the "empty" cap center.
            # The distance from origin to the plane is |n . v1|.
            # The circumcenter on the sphere is the point in the direction of the normal.
            # If we pick n such that n . v1 > 0, then the cap is centered at n.
            # The radius is arccos(n . v1).
            
            if n.dot(v1) < 0:
                n = -n
                
            d = n.dot(v1)
            # Clip for numerical stability
            d = np.clip(d, -1.0, 1.0)
            rad = np.arccos(d)
            
            if rad > max_rad:
                max_rad = rad
                worst_idx = idx
                worst_circumcenter = n
                
        return np.degrees(max_rad), worst_idx, worst_circumcenter

    current_radius, _, _ = get_covering_info(points)
    
    alpha = 0.1 # Step size
    
    while time.time() < deadline:
        current_radius, worst_idx, worst_circumcenter = get_covering_info(points)
        
        if worst_idx == -1:
            break
            
        # Get the vertices of the worst facet
        # We need to re-fetch the hull to get the simplex indices
        # To save time, we can just re-compute hull in get_covering_info and return simplex
        # Let's modify get_covering_info to return the simplex vertices
        
        # Re-implementing to get vertices
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        pts = points / norms
        try:
            hull = scipy.spatial.ConvexHull(pts)
        except:
            break
            
        max_rad = 0.0
        worst_simplex = None
        worst_circumcenter = None
        
        for idx, simplex in enumerate(hull.simplices):
            v1, v2, v3 = pts[simplex]
            n = np.cross(v1 - v2, v1 - v3)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            n = n / n_norm
            if n.dot(v1) < 0:
                n = -n
            d = np.clip(n.dot(v1), -1.0, 1.0)
            rad = np.arccos(d)
            if rad > max_rad:
                max_rad = rad
                worst_simplex = simplex
                worst_circumcenter = n
                
        if worst_simplex is None:
            break
            
        v1, v2, v3 = pts[worst_simplex]
        c = worst_circumcenter
        
        # Move v1, v2, v3 towards c
        # Tangent component of c at v: c - (v.c)v
        # Move: v + alpha * (c - (v.c)v)
        
        new_points = points.copy()
        
        for i in worst_simplex:
            v = pts[i]
            # Ensure v is normalized
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-10: continue
            v = v / v_norm
            
            proj = c - np.dot(v, c) * v
            new_v = v + alpha * proj
            new_points[i] = new_v / np.linalg.norm(new_v)
            
        # Check if this move improves the radius
        new_radius, _, _ = get_covering_info(new_points)
        
        if new_radius < current_radius:
            points = new_points
            current_radius = new_radius
        else:
            # Try a smaller step or random perturbation
            alpha *= 0.5
            if alpha < 0.001:
                # Random perturbation of the worst vertices
                for i in worst_simplex:
                    noise = np.random.normal(0, 0.01, 3)
                    new_v = points[i] + noise
                    new_points[i] = new_v / np.linalg.norm(new_v)
                new_radius, _, _ = get_covering_info(new_points)
                if new_radius < current_radius:
                    points = new_points
                    current_radius = new_radius
                alpha = 0.1 # Reset alpha

    return points
