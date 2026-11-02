import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull

N = 13

def construct():
    def objective(params):
        pts = params.reshape(N, 2)
        # Center the points to remove translation DOF
        pts = pts - np.mean(pts, axis=0)
        
        # Calculate Convex Hull Area
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume  # In 2D, volume is area
        except Exception:
            return 0.0
            
        if hull_area < 1e-10:
            return 0.0
            
        # Scale points such that hull area is 1
        scale = 1.0 / np.sqrt(hull_area)
        pts_scaled = pts * scale
        
        # Calculate minimum triangle area (Vectorized)
        idx = np.arange(N)
        i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
        mask = (i_idx < j_idx) & (j_idx < k_idx)
        i_flat = i_idx[mask]
        j_flat = j_idx[mask]
        k_flat = k_idx[mask]
        
        p_i = pts_scaled[i_flat]
        p_j = pts_scaled[j_flat]
        p_k = pts_scaled[k_flat]
        
        v1 = p_j - p_i
        v2 = p_k - p_i
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        min_area = np.min(areas)
        
        # We want to maximize min_area, so minimize -min_area
        return -min_area

    # Bounds for each coordinate: [-1, 1]
    bounds = [(-1, 1)] * (2 * N)
    
    # Run Differential Evolution
    # Using 'sobol' initialization for better coverage
    result = differential_evolution(objective, bounds, maxiter=100, popsize=15, seed=42, tol=0.01, init='sobol')
    
    # Extract best points
    best_params = result.x
    best_pts = best_params.reshape(N, 2)
    
    # Center and scale to unit hull area for consistent output
    best_pts = best_pts - np.mean(best_pts, axis=0)
    try:
        hull = ConvexHull(best_pts)
        hull_area = hull.volume
        if hull_area > 1e-10:
            best_pts = best_pts / np.sqrt(hull_area)
    except Exception:
        pass
        
    return best_pts
