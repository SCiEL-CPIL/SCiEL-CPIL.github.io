"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

# Hard deadline: 140 seconds (10s safety margin for 150s kill)
DEADLINE = 140.0
N = 50

def _covering_radius_rad(points):
    """
    Compute the covering radius (in radians) of points on the unit sphere.
    """
    # Normalize points
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    points = points / norms
    
    # Compute convex hull to find facets
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: use all pairwise distances
        dots = np.clip(points @ points.T, -1.0, 1.0)
        dists = np.arccos(dots)
        return np.max(dists)
    
    max_r = 0.0
    for simplex in hull.simplices:
        a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        idx_a, idx_b, idx_c = simplex
        
        # Compute normal to the plane
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

def construct():
    start_time = time.time()
    
    # Start from Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Optimize using Powell (does not require gradient, robust to non-smoothness)
    best_points = points.copy()
    best_radius = _covering_radius_rad(best_points)
    
    def obj(x):
        pts = x.reshape(-1, 3)
        return _covering_radius_rad(pts)
        
    # Use Powell method
    try:
        result = minimize(
            obj,
            points.flatten(),
            method='Powell',
            options={'maxiter': 200, 'maxfev': 5000}
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
    if elapsed > DEADLINE - 10:
        return best_points

    # Try additional local optimization with multiple restarts
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Perturb the best solution
        noise = np.random.randn(N, 3) * 0.05
        perturbed = best_points + noise
        # Renormalize
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        perturbed = perturbed / norms
        
        try:
            result = minimize(
                obj,
                perturbed.flatten(),
                method='Powell',
                options={'maxiter': 100, 'maxfev': 2000}
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
