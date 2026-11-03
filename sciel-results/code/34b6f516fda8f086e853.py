import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

N = 14

# EVOLVE-BLOCK-START
def _compute_score(points):
    """
    Calculate the negative of the score: min triangle area / convex hull area.
    Returns a penalty (1e6) for invalid configurations.
    """
    try:
        # Normalize points: center at origin, scale to unit mean norm
        center = np.mean(points, axis=0)
        pts = points - center
        norms = np.linalg.norm(pts, axis=1)
        mean_norm = np.mean(norms)
        if mean_norm < 1e-9:
            return 1e6 # Penalty
        
        pts = pts / mean_norm
        
        # Check for degenerate hull
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-9:
            return 1e6
            
        # Calculate min triangle area
        n = len(pts)
        
        # Vectorized approach for triangle areas
        # Get all indices i < j < k
        i, j, k = np.indices((n, n, n))
        mask = (i < j) & (j < k)
        i, j, k = i[mask], j[mask], k[mask]
        
        p1 = pts[i]
        p2 = pts[j]
        p3 = pts[k]
        
        # Cross product magnitude / 2
        v1 = p2 - p1
        v2 = p3 - p1
        areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
        
        min_area = np.min(areas)
        
        if min_area < 1e-9:
            return 1e6
            
        return -(min_area / hull_area)
        
    except Exception:
        return 1e6

def construct():
    # Differential Evolution
    # Bounds for x and y coordinates
    bounds = [(-2.0, 2.0)] * (2 * N)
    
    # DE parameters
    # popsize=10, maxiter=100, tol=1e-6
    # This should be fast enough for 28 dimensions.
    
    try:
        result = differential_evolution(
            _compute_score, 
            bounds, 
            maxiter=200, 
            popsize=15, 
            tol=1e-6, 
            seed=42,
            polish=True # Run a local optimization (Nelder-Mead) on the best result
        )
        
        best_x = result.x
        pts = best_x.reshape(N, 2)
        
        # Final normalization
        center = np.mean(pts, axis=0)
        pts = pts - center
        norms = np.linalg.norm(pts, axis=1)
        mean_norm = np.mean(norms)
        if mean_norm > 1e-9:
            pts = pts / mean_norm
            
        return pts
        
    except Exception:
        # Fallback to regular polygon if DE fails
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
# EVOLVE-BLOCK-END
