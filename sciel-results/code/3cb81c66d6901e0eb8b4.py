"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

DEADLINE = 140.0
N = 50

def _covering_radius_rad(points):
    """
    Compute the covering radius (in radians) of points on the unit sphere.
    The covering radius is the maximum circumradius of the facets of the convex hull.
    """
    # Normalize points
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    points = points / norms
    
    # Check for coincident points
    dots = points @ points.T
    np.fill_diagonal(dots, 0)
    if np.max(dots) > 1 - 1e-8:
        return np.pi  # Invalid: coincident points
    
    # Check if all points are in a closed hemisphere
    # This is equivalent to checking if the origin is inside the convex hull
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: use all pairwise distances
        dots = np.clip(points @ points.T, -1.0, 1.0)
        dists = np.arccos(dots)
        return np.max(dists)
    
    # Check if origin is inside convex hull
    # If the origin is outside, all points are in a closed hemisphere
    # We can check this by seeing if the convex hull contains the origin
    # For a convex hull in 3D, the origin is inside if and only if
    # for every facet, the origin is on the same side as the interior
    
    max_r = 0.0
    for simplex in hull.simplices:
        a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        idx_a, idx_b, idx_c = simplex
        
        # Compute normal to the plane pointing outward
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            # Degenerate triangle
            r = max(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)),
                    np.arccos(np.clip(np.dot(b, c), -1.0, 1.0)),
                    np.arccos(np.clip(np.dot(c, a), -1.0, 1.0))) / 2
            max_r = max(max_r, r)
            continue
        
        n = n / n_norm
        
        # Ensure normal points outward (away from origin)
        # The midpoint of the triangle should be on the same side as the normal
        mid = (a + b + c) / 3.0
        if np.dot(n, mid) < 0:
            n = -n
        
        # Candidate circumcenters
        c1 = n
        c2 = -n
        
        # Check c1
        dots1 = points @ c1
        max_dot1 = np.max(dots1)
        # Check if a, b, c are the nearest neighbors (max dot product)
        if (dots1[idx_a] >= max_dot1 - 1e-9 and 
            dots1[idx_b] >= max_dot1 - 1e-9 and 
            dots1[idx_c] >= max_dot1 - 1e-9 and
            abs(dots1[idx_a] - dots1[idx_b]) < 1e-9 and
            abs(dots1[idx_b] - dots1[idx_c]) < 1e-9):
            r1 = np.arccos(np.clip(dots1[idx_a], -1.0, 1.0))
            max_r = max(max_r, r1)
            
        # Check c2
        dots2 = points @ c2
        max_dot2 = np.max(dots2)
        if (dots2[idx_a] >= max_dot2 - 1e-9 and 
            dots2[idx_b] >= max_dot2 - 1e-9 and 
            dots2[idx_c] >= max_dot2 - 1e-9 and
            abs(dots2[idx_a] - dots2[idx_b]) < 1e-9 and
            abs(dots2[idx_b] - dots2[idx_c]) < 1e-9):
            r2 = np.arccos(np.clip(dots2[idx_a], -1.0, 1.0))
            max_r = max(max_r, r2)
            
    return max_r

def _fibonacci_sphere(n, offset=0.0):
    """Generate points on the sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _lloyd_iteration(points, n_iters=50, lr=0.1):
    """
    Perform Lloyd's algorithm iterations for sphere covering.
    Each point moves toward the centroid of its Voronoi cell (on the sphere).
    """
    points = points.copy()
    for it in range(n_iters):
        # For each point, find its Voronoi cell centroid
        # Approximate by finding the circumcenter of its Delaunay neighbors
        # Simpler: move each point toward the average of its k nearest neighbors
        dots = points @ points.T
        np.fill_diagonal(dots, -2)  # Exclude self
        # Find k nearest neighbors
        k = 8
        nearest_idx = np.argsort(-dots, axis=1)[:, :k]
        
        # Compute centroid of neighbors
        neighbor_points = points[nearest_idx]  # (N, k, 3)
        centroids = np.mean(neighbor_points, axis=1)
        
        # Normalize centroids
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        centroids = centroids / norms
        
        # Move points toward centroids
        points = points + lr * (centroids - points)
        
        # Renormalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        points = points / norms
        
    return points

def construct():
    start_time = time.time()
    
    # Start from multiple initializations
    inits = [
        _fibonacci_sphere(N, offset=0.0),
        _fibonacci_sphere(N, offset=np.pi / 5.0),
        _fibonacci_sphere(N, offset=2 * np.pi / 5.0),
    ]
    
    best_points = None
    best_radius = np.pi
    
    for init in inits:
        if time.time() - start_time > DEADLINE - 30:
            break
        
        # Lloyd iterations
        lloyd_pts = _lloyd_iteration(init, n_iters=80, lr=0.15)
        lloyd_rad = _covering_radius_rad(lloyd_pts)
        
        if lloyd_rad < best_radius:
            best_radius = lloyd_rad
            best_points = lloyd_pts.copy()
    
    if best_points is None:
        best_points = _fibonacci_sphere(N)
        best_radius = _covering_radius_rad(best_points)
    
    # Local optimization using Powell
    def obj(x):
        pts = x.reshape(-1, 3)
        return _covering_radius_rad(pts)
    
    try:
        result = minimize(
            obj,
            best_points.flatten(),
            method='Powell',
            options={'maxiter': 300, 'maxfev': 8000}
        )
        
        if result.fun < best_radius:
            opt_points = result.x.reshape(-1, 3)
            opt_radius = _covering_radius_rad(opt_points)
            if opt_radius < best_radius:
                best_points = opt_points
                best_radius = opt_radius
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 15:
        # Final normalization
        norms = np.linalg.norm(best_points, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        best_points = best_points / norms
        return best_points
    
    # Try additional local optimization with multiple restarts
    n_restarts = 15
    for restart in range(n_restarts):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        # Perturb the best solution
        noise = np.random.randn(N, 3) * 0.03
        perturbed = best_points + noise
        # Renormalize
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        perturbed = perturbed / norms
        
        # Quick Lloyd refinement
        lloyd_pts = _lloyd_iteration(perturbed, n_iters=30, lr=0.1)
        lloyd_rad = _covering_radius_rad(lloyd_pts)
        
        if lloyd_rad < best_radius:
            best_points = lloyd_pts.copy()
            best_radius = lloyd_rad
        
        # Powell refinement
        try:
            result = minimize(
                obj,
                lloyd_pts.flatten(),
                method='Powell',
                options={'maxiter': 150, 'maxfev': 4000}
            )
            
            if result.fun < best_radius:
                opt_points = result.x.reshape(-1, 3)
                opt_radius = _covering_radius_rad(opt_points)
                if opt_radius < best_radius:
                    best_points = opt_points
                    best_radius = opt_radius
        except Exception:
            pass
    
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    best_points = best_points / norms
    
    return best_points
# EVOLVE-BLOCK-END
