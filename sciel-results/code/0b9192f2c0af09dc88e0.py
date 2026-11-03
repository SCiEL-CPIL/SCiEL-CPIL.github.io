import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

def _compute_facet_circumradii_and_centers(points):
    """
    Compute the circumradius and circumcenter for each facet (triangle) of the convex hull.
    Returns:
        circumradii: array of shape (num_facets,) with angular circumradius of each facet
        circumcenters: array of shape (num_facets, 3) with circumcenter of each facet on unit sphere
        facet_indices: array of shape (num_facets, 3) with point indices for each facet
    """
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices  # shape (num_facets, 3)
        
        num_facets = len(simplices)
        circumradii = np.zeros(num_facets)
        circumcenters = np.zeros((num_facets, 3))
        
        for idx in range(num_facets):
            i, j, k = simplices[idx]
            p1, p2, p3 = points[i], points[j], points[k]
            
            # Compute circumcenter of triangle on sphere
            # The circumcenter is the point equidistant from all three vertices
            # For points on unit sphere, the circumcenter can be found as the normalized
            # cross product of (p2-p1) and (p3-p1), but we need to be careful.
            
            # Alternative: solve for point c such that |c-p1|=|c-p2|=|c-p3| and |c|=1
            # This is equivalent to c being in the plane of p1,p2,p3 and equidistant.
            
            # Use the formula for circumcenter of a spherical triangle
            # The circumcenter is proportional to p1*(sin(2*angle_opposite_p1)) + ... 
            # Simpler: use the fact that circumcenter is the intersection of the plane 
            # through p1,p2,p3 with the line perpendicular to the plane through the origin.
            
            # Normal to the plane
            n = np.cross(p2 - p1, p3 - p1)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-15:
                circumradii[idx] = 0
                circumcenters[idx] = p1
                continue
            n = n / n_norm
            
            # The circumcenter lies along the normal direction from the origin
            # c = t * n for some t
            # We need |c - p1| = |c - p2| = |c - p3|
            # |t*n - p1|^2 = |t*n - p2|^2
            # t^2 - 2t*(n·p1) + |p1|^2 = t^2 - 2t*(n·p2) + |p2|^2
            # Since |p1|=|p2|=1, we get n·p1 = n·p2 (which should be true)
            
            # Actually, for points on unit sphere, the circumcenter is:
            # c = (p1 + p2 + p3) / |p1 + p2 + p3| ... no that's not right either.
            
            # Let me use a more robust approach:
            # The circumcenter c satisfies: c·p1 = c·p2 = c·p3 = cos(r)
            # and |c| = 1
            # So c is the solution to: c·(p1-p2) = 0, c·(p1-p3) = 0, c·c = 1
            # This means c is parallel to the normal n, so c = ±n
            # We need to determine the sign such that c·p1 > 0 (for acute triangle) or handle obtuse
            
            # For the covering radius, we want the circumcenter that is "outside" the triangle
            # on the side away from the origin. So c = n if n·p1 > 0, else -n
            
            if np.dot(n, p1) > 0:
                c = n
            else:
                c = -n
            
            # The circumradius is arccos(c·p1)
            dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
            circumradii[idx] = np.arccos(dot_val)
            circumcenters[idx] = c
        
        return circumradii, circumcenters, simplices
        
    except Exception:
        return np.array([]), np.array([]), np.array([])

def _compute_covering_radius(points):
    """
    Computes the covering radius for points on the unit sphere.
    The covering radius is the maximum circumradius of the convex hull facets.
    """
    try:
        circumradii, _, _ = _compute_facet_circumradii_and_centers(points)
        if len(circumradii) == 0:
            return np.pi / 2.0
        return np.max(circumradii)
    except Exception:
        return np.pi / 2.0

def _facet_force_optimize(points, max_iters=200, time_limit=60.0):
    """
    Optimize points by applying forces that push points apart based on facet circumradii.
    Points in large facets are pushed outward to reduce the circumradius.
    """
    start_time = time.time()
    points = points.copy()
    
    for it in range(max_iters):
        if time.time() > start_time + time_limit:
            break
        
        circumradii, circumcenters, simplices = _compute_facet_circumradii_and_centers(points)
        if len(circumradii) == 0:
            break
        
        max_radius = np.max(circumradii)
        
        # Compute forces: for each facet, if its circumradius is above average,
        # push the three points away from the circumcenter
        avg_radius = np.mean(circumradii)
        forces = np.zeros_like(points)
        
        for idx in range(len(simplices)):
            r = circumradii[idx]
            if r > avg_radius:
                # Weight by how much above average
                weight = (r - avg_radius) / (max_radius - avg_radius + 1e-15)
                c = circumcenters[idx]
                i, j, k = simplices[idx]
                # Push points away from circumcenter (tangent to sphere)
                for p_idx in [i, j, k]:
                    p = points[p_idx]
                    # Tangential component of force away from c
                    force_dir = p - np.dot(p, c) * c
                    force_norm = np.linalg.norm(force_dir)
                    if force_norm > 1e-10:
                        force_dir /= force_norm
                        forces[p_idx] += weight * force_dir
        
        # Normalize forces to have consistent magnitude
        force_magnitudes = np.linalg.norm(forces, axis=1)
        max_force = np.max(force_magnitudes)
        if max_force > 1e-15:
            forces = forces / max_force
        
        # Update points with adaptive step size
        step_size = 0.01 * (1.0 - 0.5 * it / max_iters)
        new_points = points + step_size * forces
        
        # Project back to unit sphere
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        new_points = new_points / norms
        
        # Check if improvement
        new_radius = _compute_covering_radius(new_points)
        if new_radius < max_radius - 1e-10:
            points = new_points
        else:
            # Try smaller step
            step_size *= 0.1
            new_points = points + step_size * forces
            norms = np.linalg.norm(new_points, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            new_points = new_points / norms
            new_radius = _compute_covering_radius(new_points)
            if new_radius < max_radius - 1e-10:
                points = new_points
            else:
                # No improvement, break
                break
    
    return points

def _fibonacci_spiral(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _nelder_mead_polish(points, max_iter=1000, time_limit=30.0):
    """Final polish using Nelder-Mead on the covering radius."""
    start_time = time.time()
    
    def objective(flat_points):
        if time.time() > start_time + time_limit:
            return 10.0
        pts = flat_points.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        pts = pts / norms
        return _compute_covering_radius(pts)
    
    initial_flat = points.flatten()
    
    try:
        res = minimize(objective, initial_flat, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12, 
                               'adaptive': True, 'initial_simplex': 1e-4})
        if res.fun < _compute_covering_radius(points):
            best_flat = res.x
            norms = np.linalg.norm(best_flat.reshape(N, 3), axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return best_flat.reshape(N, 3) / norms
    except Exception:
        pass
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    # 1. Initialization: Fibonacci Spiral
    points = _fibonacci_spiral(N)
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # 2. Facet-based force optimization
    time_remaining = deadline - time.time()
    if time_remaining > 20.0:
        points = _facet_force_optimize(points, max_iters=300, time_limit=min(80.0, time_remaining - 20.0))
        current_radius = _compute_covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
    
    # 3. Final Nelder-Mead polish
    time_remaining = deadline - time.time()
    if time_remaining > 5.0:
        polished = _nelder_mead_polish(points, max_iter=500, time_limit=min(30.0, time_remaining - 5.0))
        polished_radius = _compute_covering_radius(polished)
        if polished_radius < best_radius:
            best_radius = polished_radius
            best_points = polished
    
    # Final check and return
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    best_points = best_points / norms
    
    return best_points
