"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time


def _compute_facet_circumcenters(points, hull):
    """
    For each simplex (triangle) in the hull, compute the circumcenter
    (pole of the plane) and the circumradius (angular distance to vertices).
    Returns:
        centers: array of shape (n_facets, 3)
        radii: array of shape (n_facets,)
    """
    simplices = hull.simplices
    n_facets = len(simplices)
    centers = np.zeros((n_facets, 3))
    radii = np.zeros(n_facets)
    
    for i, simplex in enumerate(simplices):
        a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        
        # Normal to the plane
        v1 = b - a
        v2 = c - a
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-15:
            # Degenerate triangle, use midpoint of longest side?
            centers[i] = (a + b + c) / 3
            norms = np.linalg.norm(centers[i])
            if norms > 1e-15:
                centers[i] /= norms
            else:
                centers[i] = a
            radii[i] = np.arccos(np.clip(np.dot(centers[i], a), -1, 1))
            continue
            
        n = cross / norm_cross
        
        # Ensure normal points outwards (away from origin)
        if np.dot(n, a) < 0:
            n = -n
            
        centers[i] = n
        
        # Circumradius is the angular distance from center to any vertex
        cos_r = np.clip(np.dot(n, a), -1.0, 1.0)
        radii[i] = np.arccos(cos_r)
        
    return centers, radii


def _refine_covering_voronoi(points, n_iterations=5000, step_size=0.05):
    """
    Refine points by moving each point towards the circumcenter of the
    worst (largest circumradius) facet that contains it.
    """
    points = points.copy()
    best_points = points.copy()
    best_radius = np.inf
    
    # Pre-compute initial radius
    try:
        hull = ConvexHull(points)
        centers, radii = _compute_facet_circumcenters(points, hull)
        best_radius = np.max(radii)
    except Exception:
        # Fallback if hull fails
        pass

    for it in range(n_iterations):
        try:
            hull = ConvexHull(points)
        except Exception:
            break
            
        centers, radii = _compute_facet_circumcenters(points, hull)
        
        # Check if we improved
        current_max_radius = np.max(radii)
        if current_max_radius < best_radius:
            best_radius = current_max_radius
            best_points = points.copy()
            
        # Map each point to the worst facet containing it
        # We need to know which facets contain which points.
        # hull.simplices is (n_facets, 3)
        
        # Initialize force/velocity for each point
        velocities = np.zeros_like(points)
        
        # For each facet, if it's the worst for any of its vertices, 
        # we want to move that vertex towards the facet center.
        # However, a vertex might be in multiple facets. We pick the one with max radius.
        
        # Create an array of the "best" (max radius) facet index for each point
        # and the corresponding center.
        # Since each point is in ~5-6 facets, we can just iterate over facets 
        # and update the "worst" record for each vertex.
        
        max_radius_for_point = np.full(N, -1.0)
        target_center_for_point = np.zeros_like(points)
        
        for f_idx in range(len(hull.simplices)):
            r = radii[f_idx]
            if r <= max_radius_for_point.max(): # Optimization: skip if not potentially worst
                 # Actually, we need to check per vertex
                 pass
                 
            simplex = hull.simplices[f_idx]
            for v_idx in simplex:
                if r > max_radius_for_point[v_idx]:
                    max_radius_for_point[v_idx] = r
                    target_center_for_point[v_idx] = centers[f_idx]
                    
        # Now, for each point, move it towards its target center
        for i in range(N):
            p = points[i]
            t = target_center_for_point[i]
            
            # Vector from p to t
            v = t - p
            v_norm = np.linalg.norm(v)
            
            if v_norm > 1e-15:
                # Move a fraction of the way
                # Use a step size that is adaptive or fixed
                # Smaller step for stability
                step = step_size * (1.0 + 0.1 * np.sin(it * 0.1)) # Slight oscillation to escape local minima?
                # Actually, fixed step is safer.
                points[i] = p + step * (v / v_norm)
                
        # Project back to sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
    return best_points, best_radius


def _fibonacci_spiral(n, offset=0.5):
    """Generate points using Fibonacci spiral."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    """Generate random points on the unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def construct():
    """Construct 130 points on the unit sphere minimizing the covering radius."""
    rng = np.random.default_rng(42)
    
    best_points = None
    best_radius = np.inf
    
    # Multiple restarts with different initializations
    for seed_offset in [0, 1, 2]:
        # Initialization: Fibonacci + Jitter
        base_points = _fibonacci_spiral(N, offset=0.5 + seed_offset * 0.1)
        # Add small random jitter to break symmetry
        jitter = rng.standard_normal((N, 3)) * 0.01
        points = base_points + jitter
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Refine using Voronoi/Lloyd-like covering minimization
        # Use a larger number of iterations
        refined_points, radius = _refine_covering_voronoi(points, n_iterations=3000, step_size=0.02)
        
        if radius < best_radius:
            best_radius = radius
            best_points = refined_points.copy()
            
    # Final fine-tuning with smaller steps
    if best_points is not None:
        final_points, final_radius = _refine_covering_voronoi(best_points, n_iterations=2000, step_size=0.005)
        if final_radius < best_radius:
            best_radius = final_radius
            best_points = final_points.copy()

    # Ensure points are on the unit sphere
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points
# EVOLVE-BLOCK-END
