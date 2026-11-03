import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _compute_covering_radius(points):
    """
    Compute the covering radius of a set of points on the unit sphere.
    The covering radius is the maximum angular distance from any point on the sphere
    to the nearest configuration point. This is equivalent to the maximum circumradius
    of the facets of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        max_radius = 0.0
        
        for tri in facets:
            p1 = points[tri[0]]
            p2 = points[tri[1]]
            p3 = points[tri[2]]
            
            # Compute the normal to the plane defined by p1, p2, p3
            v1 = p2 - p1
            v2 = p3 - p1
            cross = np.cross(v1, v2)
            norm_cross = np.linalg.norm(cross)
            
            if norm_cross < 1e-12:
                continue
                
            c = cross / norm_cross
            
            # Ensure the normal points outward (away from origin)
            if np.dot(c, p1) < 0:
                c = -c
            
            # The angular distance from the circumcenter c to the vertices is the circumradius.
            dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
            dist = np.arccos(dot_val)
            
            if dist > max_radius:
                max_radius = dist
                
    except Exception:
        return np.pi / 2
        
    return np.degrees(max_radius)

def _soft_max_objective(points, temperature=0.1):
    """
    Compute a smooth surrogate for the covering radius using log-sum-exp.
    Returns (objective_value, gradient) where objective is in radians.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        
        # Compute circumradii and their gradients
        radii = np.zeros(len(facets))
        gradients = np.zeros((N, 3))
        
        for idx, tri in enumerate(facets):
            i, j, k = tri
            p1 = points[i]
            p2 = points[j]
            p3 = points[k]
            
            v1 = p2 - p1
            v2 = p3 - p1
            cross = np.cross(v1, v2)
            norm_cross = np.linalg.norm(cross)
            
            if norm_cross < 1e-12:
                radii[idx] = 0.0
                continue
                
            c = cross / norm_cross
            
            if np.dot(c, p1) < 0:
                c = -c
            
            dot_val = np.clip(np.dot(c, p1), -1.0 + 1e-10, 1.0 - 1e-10)
            dist = np.arccos(dot_val)
            radii[idx] = dist
            
            # Gradient of arccos(dot(c, p1)) with respect to p1, p2, p3
            # d/dp arccos(c·p) = -c / sqrt(1 - (c·p)^2)
            denom = np.sqrt(np.maximum(1.0 - dot_val**2, 1e-12))
            grad_p = -c / denom
            
            gradients[i] += grad_p
            gradients[j] += grad_p
            gradients[k] += grad_p
        
        # Log-sum-exp for soft max
        max_r = np.max(radii)
        if max_r < 1e-10:
            return 0.0, gradients
        
        shifted = radii - max_r
        log_sum_exp = max_r + np.log(np.sum(np.exp(shifted / temperature))) * temperature
        
        # Gradient of log-sum-exp: (1/temperature) * exp((r_i - max_r)/temperature) * dr_i
        weights = np.exp(shifted / temperature) / np.sum(np.exp(shifted / temperature))
        grad_obj = np.zeros((N, 3))
        for idx in range(len(facets)):
            grad_obj += weights[idx] * gradients[idx]
        
        return log_sum_exp, grad_obj
    except Exception:
        return np.pi / 2, np.zeros((N, 3))

def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return points / norms

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return points

def _lloyd_iteration(points):
    """
    Perform one iteration of Lloyd's algorithm.
    1. Compute convex hull.
    2. For each point, find the circumcenters of the triangles incident to it.
    3. Move the point to the normalized sum of these circumcenters.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        
        # Initialize sum of circumcenters for each point
        sums = np.zeros((N, 3))
        counts = np.zeros(N)
        
        for tri in facets:
            i, j, k = tri
            p1 = points[i]
            p2 = points[j]
            p3 = points[k]
            
            v1 = p2 - p1
            v2 = p3 - p1
            cross = np.cross(v1, v2)
            norm_cross = np.linalg.norm(cross)
            
            if norm_cross < 1e-12:
                continue
                
            c = cross / norm_cross
            
            if np.dot(c, p1) < 0:
                c = -c
            
            # This circumcenter c is a vertex of the Voronoi cell for i, j, and k
            sums[i] += c
            sums[j] += c
            sums[k] += c
            
            counts[i] += 1
            counts[j] += 1
            counts[k] += 1
        
        new_points = points.copy()
        for i in range(N):
            if counts[i] > 0:
                # Move to the normalized sum of the circumcenters
                # This approximates the centroid of the Voronoi cell
                new_points[i] = sums[i]
                norm = np.linalg.norm(new_points[i])
                if norm > 1e-12:
                    new_points[i] /= norm
                else:
                    # Fallback if sum is zero (should not happen for well-distributed points)
                    new_points[i] = points[i]
            
        return new_points
    except Exception:
        return points

def _gradient_descent(points, start_time, deadline, initial_lr=0.005, min_lr=1e-6, temperature=0.05):
    """
    Optimize using gradient descent on a smooth surrogate objective.
    Uses log-sum-exp for soft-max of circumradii.
    """
    current_points = points.copy()
    best_points = current_points.copy()
    best_radius = _compute_covering_radius(current_points)
    lr = initial_lr
    
    while time.time() < deadline:
        obj_val, grad = _soft_max_objective(current_points, temperature)
        
        # Project gradient onto tangent space (remove radial component)
        for i in range(N):
            radial = np.dot(grad[i], current_points[i])
            grad[i] -= radial * current_points[i]
        
        # Take gradient step
        new_points = current_points - lr * grad
        new_points = _project_to_sphere(new_points)
        
        current_radius = _compute_covering_radius(new_points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = new_points.copy()
            current_points = new_points.copy()
            # Decay learning rate slowly
            lr *= 0.999
        else:
            # Reduce learning rate and try again
            lr *= 0.9
            if lr < min_lr:
                break
        
        # Check time
        if time.time() > deadline - 5:
            break
    
    return best_points, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # 8s safety margin
    
    # Start with Fibonacci spiral
    points = _fibonacci_spiral(N)
    points = _project_to_sphere(points)
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Phase 1: Run Lloyd's algorithm for initial convergence
    lloyd_deadline = start_time + 60.0
    stall_count = 0
    max_stall = 15
    
    while time.time() < lloyd_deadline:
        new_points = _lloyd_iteration(points)
        new_points = _project_to_sphere(new_points)
        
        current_radius = _compute_covering_radius(new_points)
        
        if current_radius < best_radius - 1e-10:
            best_radius = current_radius
            best_points = new_points.copy()
            points = new_points.copy()
            stall_count = 0
        else:
            points = new_points.copy()
            stall_count += 1
            
            if stall_count > max_stall:
                # Perturb a random subset
                num_perturb = max(5, N // 10)
                indices = np.random.choice(N, num_perturb, replace=False)
                noise = np.random.normal(0, 0.02, (num_perturb, 3))
                points[indices] += noise
                points[indices] = _project_to_sphere(points[indices])
                stall_count = 0
    
    # Phase 2: Gradient descent with smooth surrogate
    grad_deadline = start_time + 120.0
    points = best_points.copy()
    gd_points, gd_radius = _gradient_descent(points, start_time, grad_deadline, 
                                              initial_lr=0.003, min_lr=1e-7, temperature=0.03)
    
    if gd_radius < best_radius:
        best_radius = gd_radius
        best_points = gd_points.copy()
    
    # Phase 3: Final high-precision polish with Lloyd's
    polish_deadline = start_time + 140.0
    points = best_points.copy()
    stall_count = 0
    max_stall = 8
    
    while time.time() < polish_deadline:
        new_points = _lloyd_iteration(points)
        new_points = _project_to_sphere(new_points)
        
        current_radius = _compute_covering_radius(new_points)
        
        if current_radius < best_radius - 1e-10:
            best_radius = current_radius
            best_points = new_points.copy()
            points = new_points.copy()
            stall_count = 0
        else:
            stall_count += 1
            if stall_count > max_stall:
                break
    
    return best_points
