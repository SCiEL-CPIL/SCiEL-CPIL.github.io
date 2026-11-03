import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _compute_covering_radius(points):
    """
    Compute the covering radius of a set of points on the unit sphere.
    Returns radius in degrees.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        max_radius = 0.0
        
        for tri in facets:
            p1 = points[tri[0]]
            p2 = points[tri[1]]
            p3 = points[tri[2]]
            
            v1 = p2 - p1
            v2 = p3 - p1
            cross = np.cross(v1, v2)
            norm_cross = np.linalg.norm(cross)
            
            if norm_cross < 1e-12:
                continue
                
            c = cross / norm_cross
            
            dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
            dist = np.arccos(dot_val)
            
            if dist > np.pi / 2:
                dist = np.pi - dist
                
            if dist > max_radius:
                max_radius = dist
                
    except Exception:
        return 180.0
        
    return np.degrees(max_radius)

def _compute_all_facet_data(points):
    """
    Compute circumcenters and radii for all facets.
    Returns list of (circumcenter_vector, radius_radians, facet_indices)
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return []
        
    facets = hull.simplices
    results = []
    
    for tri in facets:
        p1 = points[tri[0]]
        p2 = points[tri[1]]
        p3 = points[tri[2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-12:
            continue
            
        c = cross / norm_cross
        
        dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
        dist = np.arccos(dot_val)
        
        # Ensure c points towards the "hole" (the empty cap center)
        # For a convex hull, the outward normal points away from the interior.
        # The "hole" is the region of the sphere not covered by the caps of the vertices.
        # If the triangle is small, the circumcenter is the outward normal.
        # If the triangle is large (covering > hemisphere), the circumcenter of the *empty* part
        # is the inward normal.
        # However, for a good packing, all facets are small.
        # The covering radius is determined by the maximum distance to the nearest point.
        # This maximum occurs at a circumcenter of a Delaunay triangle.
        # For small triangles, the circumcenter is c (outward normal).
        # The distance from c to the vertices is dist.
        # We want to minimize this dist.
        
        # Note: if dist > pi/2, the "hole" is actually on the other side?
        # No, the circumcenter of the spherical triangle is unique.
        # If the triangle contains the origin in its convex hull (in 3D), i.e., is large,
        # then the circumcenter on the sphere is far.
        # But we are minimizing the max distance.
        # So we just use dist.
        
        results.append((c, dist, tri))
        
    return results

def construct():
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    rng = np.random.default_rng(42)
    
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Phase 1: Quick Gradient Descent to settle into a local minimum
    current_radius = _compute_covering_radius(points)
    best_radius = current_radius
    best_points = points.copy()
    
    step_size = 0.05
    grad_deadline = start_time + 20.0
    
    while time.time() < grad_deadline:
        # Compute gradients for worst facets
        try:
            hull = ConvexHull(points)
            facets = hull.simplices
            max_radius_rad = 0
            worst_facets = []
            
            for tri in facets:
                p1 = points[tri[0]]
                p2 = points[tri[1]]
                p3 = points[tri[2]]
                
                v1 = p2 - p1
                v2 = p3 - p1
                cross = np.cross(v1, v2)
                norm_cross = np.linalg.norm(cross)
                
                if norm_cross < 1e-12: continue
                c = cross / norm_cross
                dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
                dist = np.arccos(dot_val)
                if dist > np.pi/2: dist = np.pi - dist
                
                if dist > max_radius_rad:
                    max_radius_rad = dist
                    worst_facets = [tri]
                elif abs(dist - max_radius_rad) < 1e-6:
                    worst_facets.append(tri)
            
            gradient = np.zeros_like(points)
            # Use top 5 worst facets
            for tri in worst_facets[:5]:
                p1 = points[tri[0]]
                p2 = points[tri[1]]
                p3 = points[tri[2]]
                v1 = p2 - p1
                v2 = p3 - p1
                cross = np.cross(v1, v2)
                norm_cross = np.linalg.norm(cross)
                if norm_cross < 1e-12: continue
                c = cross / norm_cross
                
                for idx in tri:
                    p_i = points[idx]
                    dot_pc = np.clip(np.dot(p_i, c), -1.0, 1.0)
                    denom = np.sqrt(1.0 - dot_pc * dot_pc)
                    if denom > 1e-9:
                        gradient[idx] += -c / denom
            
            grad_norm = np.linalg.norm(gradient)
            if grad_norm > 1e-9:
                gradient /= grad_norm
                new_points = points + step_size * gradient
                new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
                new_radius = _compute_covering_radius(new_points)
                
                if new_radius < current_radius:
                    points = new_points
                    current_radius = new_radius
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_points = points.copy()
                    step_size *= 0.95
                else:
                    step_size *= 0.9
            else:
                break
        except Exception:
            break

    # Phase 2: Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.9995
    noise_scale = 0.05
    
    while time.time() < deadline:
        # Pick a random point
        idx = rng.integers(0, N)
        
        # Generate a random move
        noise = rng.normal(0, noise_scale, 3)
        new_p = points[idx] + noise
        norm_new_p = np.linalg.norm(new_p)
        if norm_new_p < 1e-9:
            continue
        new_p /= norm_new_p
        
        # Temporarily update
        old_p = points[idx].copy()
        points[idx] = new_p
        
        new_radius = _compute_covering_radius(points)
        delta_R = new_radius - current_radius
        
        # Accept if better or with probability exp(-delta/T)
        if delta_R < 0:
            accept = True
        elif delta_R == 0:
            accept = True
        else:
            r = rng.random()
            if r < 1e-15: r = 1e-15
            accept = np.log(r) < -delta_R / T
        
        if accept:
            current_radius = new_radius
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
        else:
            # Revert
            points[idx] = old_p
            
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min

    return best_points
