# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull

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

def compute_covering_radius(points):
    """
    Compute the covering radius in radians.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: if hull fails, assume bad configuration
        return np.pi / 2.0 

    max_radius = 0.0
    for simplex in hull.simplices:
        a, b, c = points[simplex]
        # Circumcenter of triangle abc on the sphere
        # o is parallel to (a-b) x (b-c)
        v1 = a - b
        v2 = b - c
        normal = np.cross(v1, v2)
        norm_n = np.linalg.norm(normal)
        if norm_n < 1e-10:
            # Degenerate, points nearly collinear
            # Use midpoint of longest edge as approx center? 
            # Or just skip? If collinear, the "facet" is a line, 
            # the cap is determined by the endpoints.
            # For a line segment on sphere, the covering radius is half the arc length?
            # No, the facet is part of the hull. If 3 points are collinear on a great circle, 
            # they don't form a triangular facet in the convex hull sense (they are coplanar with origin).
            # ConvexHull usually handles this by not including degenerate facets or 
            # including them with zero area. 
            # If norm is small, the triangle is flat.
            continue
        
        o = normal / norm_n
        # Ensure o is on the correct side (inside the hull)
        centroid = (a + b + c) / 3.0
        if np.dot(o, centroid) < 0:
            o = -o
            
        cos_r = np.clip(np.dot(o, a), -1.0, 1.0)
        r = np.arccos(cos_r)
        if r > max_radius:
            max_radius = r
            
    return max_radius

def construct():
    # Initial configuration
    points = fibonacci_sphere(N)
    points = normalize(points)
    
    best_points = points.copy()
    best_radius = compute_covering_radius(best_points)
    
    # Simulated Annealing parameters
    T_start = 0.1 # Initial temperature (in radians squared? or just scale)
    T_end = 1e-6
    steps = 5000
    step_size = 0.1 # Initial perturbation size
    
    # Cooling schedule
    # T = T_start * (T_end / T_start) ** (i / steps)
    
    current_radius = best_radius
    
    rng = np.random.RandomState(42)
    
    for i in range(steps):
        # Cooling
        progress = i / steps
        T = T_start * (T_end / T_start) ** progress
        
        # Generate move
        idx = rng.randint(N)
        # Random direction
        noise = rng.randn(3)
        # New point
        new_point = points[idx] + step_size * noise
        new_point /= np.linalg.norm(new_point)
        
        # Apply move temporarily
        old_point = points[idx].copy()
        points[idx] = new_point
        
        # Evaluate
        new_radius = compute_covering_radius(points)
        
        # Accept/Reject
        delta = new_radius - current_radius
        if delta < 0 or rng.rand() < np.exp(-delta / T):
            current_radius = new_radius
            if new_radius < best_radius:
                best_radius = new_radius
                best_points = points.copy()
        else:
            # Revert
            points[idx] = old_point
            
        # Adaptive step size (optional, but helps)
        # If we accept too many, increase step. If too few, decrease.
        # We don't track acceptance rate easily here without extra vars.
        # Let's just keep step_size constant or decrease it slowly.
        step_size = 0.1 * (1.0 - 0.5 * progress)
        
    # Final polish: local search on best_points
    # Use a simple gradient-like step: move points in worst facet towards center
    # This is a local refinement, distinct from the global SA search.
    # We will run a few iterations of this.
    
    for _ in range(100):
        # Find worst facet
        try:
            hull = ConvexHull(best_points)
        except:
            break
            
        max_r = 0
        worst_simplex = None
        for simplex in hull.simplices:
            a, b, c = best_points[simplex]
            v1 = a - b
            v2 = b - c
            normal = np.cross(v1, v2)
            norm_n = np.linalg.norm(normal)
            if norm_n < 1e-10: continue
            o = normal / norm_n
            centroid = (a + b + c) / 3.0
            if np.dot(o, centroid) < 0: o = -o
            cos_r = np.clip(np.dot(o, a), -1.0, 1.0)
            r = np.arccos(cos_r)
            if r > max_r:
                max_r = r
                worst_simplex = simplex
        
        if worst_simplex is None: break
        
        # Move points in worst simplex towards center
        a, b, c = best_points[worst_simplex]
        v1 = a - b
        v2 = b - c
        normal = np.cross(v1, v2)
        o = normal / np.linalg.norm(normal)
        centroid = (a + b + c) / 3.0
        if np.dot(o, centroid) < 0: o = -o
        
        for idx in worst_simplex:
            best_points[idx] = best_points[idx] + 0.01 * o
            best_points[idx] /= np.linalg.norm(best_points[idx])
            
    return best_points
# EVOLVE-BLOCK-END
