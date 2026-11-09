"""Model-and-optimize approach for covering the sphere with 130 equal caps."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 130

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return points / norms

def _covering_radius(points):
    """
    Compute the covering radius in radians.
    points: (N, 3) unit vectors.
    The covering radius is the maximum circumradius of the facets
    of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        max_circumradius = 0.0
        for simplex in hull.simplices:
            # simplex contains 3 indices for a triangle
            tri = points[simplex]
            a, b, c = tri[0], tri[1], tri[2]
            
            # The circumcenter x satisfies:
            # x·a = x·b = x·c = cos(R)
            # and |x| = 1
            # x is in the intersection of planes x·(a-b)=0 and x·(b-c)=0
            # This gives x ∝ (a-b) × (b-c)
            
            v1 = a - b
            v2 = b - c
            x_dir = np.cross(v1, v2)
            x_norm = np.linalg.norm(x_dir)
            if x_norm < 1e-10:
                continue
            x = x_dir / x_norm
            
            # Check if x is on the same side as the triangle centroid
            centroid = (a + b + c) / 3.0
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 1e-10:
                centroid = centroid / centroid_norm
                if np.dot(x, centroid) < 0:
                    x = -x
            
            # Compute the circumradius
            cos_R = np.clip(np.dot(x, a), -1.0, 1.0)
            R = np.arccos(cos_R)
            
            max_circumradius = max(max_circumradius, R)
        
        return max_circumradius
    except Exception:
        # Fallback: use a grid-based approximation
        theta = np.linspace(0, 2*np.pi, 100)
        phi = np.linspace(0, np.pi, 100)
        TH, PH = np.meshgrid(theta, phi)
        X = np.sin(PH) * np.cos(TH)
        Y = np.sin(PH) * np.sin(TH)
        Z = np.cos(PH)
        grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        dots = grid_points @ points.T
        cos_angles = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(cos_angles)
        min_angles = np.min(angles, axis=1)
        return np.max(min_angles)

def _lloyd_iteration(points, rng):
    """
    One iteration of Lloyd's algorithm for spherical codes.
    Move each point to the centroid of its Voronoi cell (spherical).
    """
    n = len(points)
    new_points = np.zeros_like(points)
    
    # For each point, find its Voronoi cell on the sphere
    # Approximate by finding the centroid of nearby points within some radius
    # Better: use the fact that the Voronoi cell of point i consists of all
    # sphere points closer to i than to any other point
    
    # Efficient approach: for each point, compute the spherical centroid of
    # all grid points in its Voronoi cell
    
    # Use a fine grid to approximate Voronoi cells
    theta = np.linspace(0, 2*np.pi, 200)
    phi = np.linspace(0, np.pi, 200)
    TH, PH = np.meshgrid(theta, phi)
    X = np.sin(PH) * np.cos(TH)
    Y = np.sin(PH) * np.sin(TH)
    Z = np.cos(PH)
    grid = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # For each grid point, find nearest configuration point
    dots = grid @ points.T  # (num_grid, N)
    nearest = np.argmax(dots, axis=1)
    
    # For each configuration point, compute centroid of its Voronoi cell
    for i in range(n):
        mask = (nearest == i)
        if np.sum(mask) > 0:
            cell_points = grid[mask]
            centroid = np.mean(cell_points, axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 1e-10:
                new_points[i] = centroid / centroid_norm
            else:
                new_points[i] = points[i]
        else:
            new_points[i] = points[i]
    
    return new_points

def _lloyd_optimize(points, max_iter=100, time_limit=80.0):
    """
    Run Lloyd's algorithm to optimize the configuration.
    """
    start_time = time.time()
    rng = np.random.default_rng(42)
    current = points.copy()
    best = current.copy()
    best_val = _covering_radius(best)
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Lloyd iteration
        new_pts = _lloyd_iteration(current, rng)
        
        # Check if we made progress
        new_val = _covering_radius(new_pts)
        
        if new_val < best_val:
            best = new_pts.copy()
            best_val = new_val
        
        # Check convergence
        if np.max(np.linalg.norm(new_pts - current, axis=1)) < 1e-6:
            break
        
        current = new_pts
    
    return best

def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _polish(points, time_limit=10.0):
    """
    Final polish using L-BFGS-B with tighter tolerance.
    """
    start_time = time.time()
    
    def flat_objective(flat):
        pts = _normalize(flat)
        return _covering_radius(pts)
    
    best = points.copy()
    best_val = _covering_radius(best)
    
    try:
        result = minimize(
            flat_objective,
            points.ravel(),
            method='L-BFGS-B',
            options={'maxiter': 100, 'maxfun': 200, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if result.fun < best_val:
            best = _normalize(result.x)
            best_val = result.fun
    except Exception:
        pass
    
    return best

def construct():
    """Construct 130 points on the unit sphere to minimize covering radius."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 150s limit, leave 10s buffer
    
    # Start with Fibonacci sphere
    points = _fibonacci_sphere(N)
    
    # Lloyd's algorithm optimization
    if time.time() < deadline - 20:
        points = _lloyd_optimize(points, max_iter=100, time_limit=80.0)
    
    # Final polish
    if time.time() < deadline - 5:
        points = _polish(points, time_limit=10.0)
    
    # Final normalization
    points = _normalize(points)
    
    # Verify no coincident points
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(points[i] - points[j]) < 1e-8:
                points[j] = _normalize(points[j] + np.random.normal(0, 0.01, 3))
    
    return points
