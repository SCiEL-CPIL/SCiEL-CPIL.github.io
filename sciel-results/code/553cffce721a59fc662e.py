import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130
TIME_LIMIT = 135.0  # safety margin before 150s kill

def _covering_radius(points):
    """Compute the exact covering radius from the convex hull."""
    try:
        hull = ConvexHull(points)
        max_r = 0.0
        for simplex in hull.simplices:
            v1 = points[simplex[0]]
            v2 = points[simplex[1]]
            v3 = points[simplex[2]]
            
            # Circumcenter of the spherical triangle
            # The circumcenter lies along the normal to the plane containing v1, v2, v3
            n = np.cross(v1, v2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-12:
                continue
            n = n / norm_n
            
            # Check if the circumcenter is on the correct side
            # For a convex hull facet, the normal points outward
            # The circumcenter should be "outside" the triangle in the sense of the cap
            d1 = np.dot(n, v1)
            d2 = np.dot(n, v2)
            d3 = np.dot(n, v3)
            
            # If any dot product is negative, try -n
            if d1 < 0 or d2 < 0 or d3 < 0:
                n = -n
                d1, d2, d3 = -d1, -d2, -d3
            
            # Compute distances
            r1 = np.arccos(np.clip(d1, -1, 1))
            
            # Check if circumcenter is inside the spherical triangle
            # A point c is inside if it's on the same side of each edge as the opposite vertex
            n12 = np.cross(v1, v2)
            n23 = np.cross(v2, v3)
            n31 = np.cross(v3, v1)
            
            inside = True
            if np.dot(n12, v3) * np.dot(n12, n) < 0:
                inside = False
            if np.dot(n23, v1) * np.dot(n23, n) < 0:
                inside = False
            if np.dot(n31, v2) * np.dot(n31, n) < 0:
                inside = False
            
            if inside:
                r_facet = r1
            else:
                # Max distance is to the midpoint of the longest edge
                d12 = np.arccos(np.clip(np.dot(v1, v2), -1, 1))
                d23 = np.arccos(np.clip(np.dot(v2, v3), -1, 1))
                d31 = np.arccos(np.clip(np.dot(v3, v1), -1, 1))
                max_edge = max(d12, d23, d31)
                r_facet = max_edge / 2.0
            
            if r_facet > max_r:
                max_r = r_facet
        return max_r
    except Exception:
        return np.pi

def _soft_covering_radius_and_grad(points, beta=50.0):
    """
    Compute a smooth approximation of the covering radius and its gradient.
    Uses log-sum-exp of squared distances to approximate max distance.
    
    The key insight: the covering radius is determined by the Voronoi vertices,
    which are circumcenters of hull facets. We approximate this by considering
    the distance from each point to the "worst" direction in its Voronoi cell.
    
    A simpler approach: use the fact that the covering radius equals the max over
    all directions y of min_i angle(y, p_i). We approximate this with a smooth
    function based on the minimum distance to each point.
    """
    # For a smooth approximation, we use:
    # f(points) = (1/beta) * log(sum_j exp(-beta * d_j^2))
    # where d_j is the distance from the j-th Voronoi vertex to its nearest point
    # But we don't have Voronoi vertices easily.
    
    # Alternative: use the maximum pairwise angular distance as a proxy
    # This is not exactly the covering radius but is related.
    
    # Better approach: compute the covering radius via hull, but use
    # a smooth surrogate for optimization.
    
    # Let's use a different strategy: optimize the maximum of the
    # circumradii of all hull facets using a smooth approximation.
    
    # For now, return the exact covering radius (no gradient)
    # and use scipy's minimize with numerical gradients.
    cr = _covering_radius(points)
    return cr, None

def _fibonacci_sphere(n):
    """Generate points on a sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, rng):
    """Generate random points on the unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _project_to_sphere(pts):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _optimize_from_start(start_pts, time_budget):
    """Optimize starting from a given configuration."""
    start_time = time.time()
    pts = start_pts.copy()
    
    # Define objective function for scipy
    def objective(x):
        p = _project_to_sphere(x.reshape(-1, 3))
        return _covering_radius(p)
    
    # Use Nelder-Mead or similar derivative-free method
    # Since we have 390 parameters, this is expensive.
    # Instead, use a custom gradient-free optimization.
    
    # Custom optimization: move points towards the worst facet's circumcenter
    best_pts = pts.copy()
    best_radius = _covering_radius(pts)
    
    step_size = 0.02
    no_improve = 0
    max_no_improve = 50
    
    while time.time() - start_time < time_budget:
        # Find the worst facet
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        worst_r = 0
        worst_c = None
        worst_v = None
        
        for simplex in hull.simplices:
            v1 = pts[simplex[0]]
            v2 = pts[simplex[1]]
            v3 = pts[simplex[2]]
            
            n = np.cross(v1, v2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-12:
                continue
            n = n / norm_n
            
            d1 = np.dot(n, v1)
            d2 = np.dot(n, v2)
            d3 = np.dot(n, v3)
            
            if d1 < 0 or d2 < 0 or d3 < 0:
                n = -n
                d1, d2, d3 = -d1, -d2, -d3
            
            r1 = np.arccos(np.clip(d1, -1, 1))
            
            n12 = np.cross(v1, v2)
            n23 = np.cross(v2, v3)
            n31 = np.cross(v3, v1)
            
            inside = True
            if np.dot(n12, v3) * np.dot(n12, n) < 0:
                inside = False
            if np.dot(n23, v1) * np.dot(n23, n) < 0:
                inside = False
            if np.dot(n31, v2) * np.dot(n31, n) < 0:
                inside = False
            
            if inside:
                r_facet = r1
            else:
                d12 = np.arccos(np.clip(np.dot(v1, v2), -1, 1))
                d23 = np.arccos(np.clip(np.dot(v2, v3), -1, 1))
                d31 = np.arccos(np.clip(np.dot(v3, v1), -1, 1))
                max_edge = max(d12, d23, d31)
                r_facet = max_edge / 2.0
            
            if r_facet > worst_r:
                worst_r = r_facet
                worst_c = n
                worst_v = simplex
        
        if worst_c is None:
            break
        
        # Move the three vertices of the worst facet towards the circumcenter
        new_pts = pts.copy()
        for idx in worst_v:
            old = new_pts[idx]
            new = old + step_size * (worst_c - old)
            new_pts[idx] = new / np.linalg.norm(new)
        
        new_radius = _covering_radius(new_pts)
        
        if new_radius < best_radius - 1e-10:
            pts = new_pts
            best_pts = pts.copy()
            best_radius = new_radius
            no_improve = 0
            step_size *= 0.98
        else:
            no_improve += 1
            step_size *= 0.95
        
        if no_improve > max_no_improve:
            step_size *= 0.5
            no_improve = 0
            if step_size < 1e-4:
                break
    
    return best_pts, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    rng = np.random.RandomState(42)
    
    # Try multiple starting configurations
    best_pts = None
    best_radius = np.inf
    
    # 1. Fibonacci spiral
    start_configs = [_fibonacci_sphere(N)]
    
    # 2. Random starts
    for _ in range(5):
        start_configs.append(_random_sphere(N, rng))
    
    # 3. Slightly perturbed Fibonacci
    fib = _fibonacci_sphere(N)
    for _ in range(3):
        perturbed = fib + rng.standard_normal((N, 3)) * 0.1
        start_configs.append(_project_to_sphere(perturbed))
    
    for i, start_pts in enumerate(start_configs):
        if time.time() > deadline - 10:
            break
        
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        # Allocate time based on number of configs
        time_per_config = min(20.0, remaining / max(1, len(start_configs) - i))
        
        pts, radius = _optimize_from_start(start_pts, time_per_config)
        
        if radius < best_radius:
            best_radius = radius
            best_pts = pts.copy()
    
    # Final polish: local optimization from best result
    if best_pts is not None and time.time() < deadline - 5:
        remaining = deadline - time.time()
        best_pts, best_radius = _optimize_from_start(best_pts, remaining)
    
    # Ensure points are on the unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    return best_pts
