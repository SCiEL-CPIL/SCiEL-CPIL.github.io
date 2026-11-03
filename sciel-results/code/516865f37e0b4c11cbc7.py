import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return pts / norms

def _circumcenter_robust(p1, p2, p3):
    """Compute circumcenter of triangle on unit sphere, robust to degeneracy."""
    # Try multiple edge pairs for cross product to handle near-degenerate triangles
    edges = [
        (p2 - p1, p3 - p1),
        (p1 - p2, p3 - p2),
        (p1 - p3, p2 - p3)
    ]
    
    for v1, v2 in edges:
        n = np.cross(v1, v2)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        
        n = n / n_norm
        # The circumcenter should be along the normal direction
        # For points on unit sphere, circumcenter is the normalized normal
        # if the triangle is acute, otherwise it's on the opposite side
        center = n.copy()
        # Check if center is on the same side as the triangle
        dot1 = np.dot(center, p1)
        dot2 = np.dot(center, p2)
        dot3 = np.dot(center, p3)
        
        # If all dots are positive, center is on the same side
        if dot1 > 0 and dot2 > 0 and dot3 > 0:
            return center
        else:
            return -center
    
    # Fallback: average of the three points, normalized
    center = (p1 + p2 + p3) / 3.0
    norm = np.linalg.norm(center)
    if norm < 1e-10:
        return p1.copy()
    return center / norm

def _covering_radius_deg(points):
    """Compute the covering radius in degrees using convex hull."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return 180.0  # Invalid hull, return large value
        
    max_radius = 0.0
    simplices = hull.simplices
    
    for simplex in simplices:
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]
        
        # Compute circumcenter
        center = _circumcenter_robust(p1, p2, p3)
        
        # The covering radius for this facet is the angular distance from center to any vertex
        # (they should all be equal for a valid circumcenter)
        dist1 = np.arccos(np.clip(np.dot(center, p1), -1.0, 1.0))
        max_radius = max(max_radius, dist1)
    
    return np.degrees(max_radius)

def _fibonacci_spiral(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _optimize(pts, deadline):
    """Optimize points by moving vertices of largest facet toward circumcenter."""
    current_pts = pts.copy()
    best_pts = pts.copy()
    best_radius = _covering_radius_deg(pts)
    
    # Adaptive step size
    step_size = 0.1
    no_improve_count = 0
    max_no_improve = 10
    
    while time.time() < deadline:
        # Find the facet with the largest circumradius
        try:
            hull = ConvexHull(current_pts)
        except Exception:
            break
            
        max_radius = 0.0
        max_simplex = None
        max_center = None
        
        for simplex in hull.simplices:
            p1 = current_pts[simplex[0]]
            p2 = current_pts[simplex[1]]
            p3 = current_pts[simplex[2]]
            
            center = _circumcenter_robust(p1, p2, p3)
            dist = np.arccos(np.clip(np.dot(center, p1), -1.0, 1.0))
            
            if dist > max_radius:
                max_radius = dist
                max_simplex = simplex
                max_center = center
        
        if max_simplex is None:
            break
            
        # Move the three vertices toward the circumcenter
        new_pts = current_pts.copy()
        for idx in max_simplex:
            p = current_pts[idx]
            # Move toward center with step size
            direction = max_center - p
            dir_norm = np.linalg.norm(direction)
            if dir_norm < 1e-10:
                continue
            direction = direction / dir_norm
            new_pts[idx] = p + step_size * direction
            new_pts[idx] = new_pts[idx] / np.linalg.norm(new_pts[idx])
        
        # Evaluate new configuration
        new_radius = _covering_radius_deg(new_pts)
        
        if new_radius < best_radius - 1e-8:
            best_radius = new_radius
            best_pts = new_pts.copy()
            current_pts = new_pts.copy()
            no_improve_count = 0
            # Slightly increase step size on improvement
            step_size = min(step_size * 1.05, 0.3)
        else:
            no_improve_count += 1
            # Decrease step size when no improvement
            step_size = max(step_size * 0.9, 0.01)
            
            if no_improve_count >= max_no_improve:
                # Try a small random perturbation to escape local minimum
                if time.time() < deadline - 5.0:
                    current_pts = best_pts.copy()
                    # Perturb a random subset of points
                    n_perturb = max(3, N // 10)
                    idxs = np.random.choice(N, n_perturb, replace=False)
                    for idx in idxs:
                        # Random direction on tangent plane
                        p = current_pts[idx]
                        rand_vec = np.random.randn(3)
                        tangent = rand_vec - np.dot(rand_vec, p) * p
                        tangent_norm = np.linalg.norm(tangent)
                        if tangent_norm < 1e-10:
                            continue
                        tangent = tangent / tangent_norm
                        current_pts[idx] = p + 0.05 * tangent
                        current_pts[idx] = current_pts[idx] / np.linalg.norm(current_pts[idx])
                    
                    no_improve_count = 0
                    step_size = 0.1
                else:
                    break
    
    return best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Initialize with Fibonacci spiral
    pts = _fibonacci_spiral(N)
    
    # Optimize using facet-based approach
    try:
        optimized_pts = _optimize(pts, deadline)
        if np.all(np.isfinite(optimized_pts)):
            return _normalize(optimized_pts)
    except Exception as e:
        pass
    
    # Fallback to initial points if optimization fails
    return _normalize(pts)
