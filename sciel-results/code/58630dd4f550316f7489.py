# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices using optimization.
    
    Strategy:
    Parameterize the polygon by 20 radii r_i for fixed angles theta_i = 2*pi*i/20.
    This ensures the vertices are in cyclic order and close to a regular polygon.
    We use Nelder-Mead to maximize the minimum vertex score defined in the task.
    A penalty is added if the polygon is not strictly convex or normalization fails.
    """
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Initial guess: regular polygon (r=1)
    r_init = np.ones(n)
    
    def calculate_score(r):
        """
        Calculate the score for a given set of radii.
        Returns negative score for minimization (since we want to maximize score).
        """
        # Construct vertices
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        polygon = np.stack([x, y], axis=1)
        
        # Normalize
        centroid = np.mean(polygon, axis=0)
        polygon_norm = polygon - centroid
        dists_centroid = np.linalg.norm(polygon_norm, axis=1)
        mean_dist = np.mean(dists_centroid)
        
        if mean_dist <= 0.01:
            return 1e6 # High penalty
        
        polygon_norm = polygon_norm / mean_dist
        
        # Check convexity
        # Cross product of consecutive edges should have same sign
        # v_i = p_{i+1} - p_i
        # cross = v_i x v_{i+1}
        edges = np.roll(polygon_norm, -1, axis=0) - polygon_norm
        cross_products = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
        
        # Strictly convex: all cross products non-zero and same sign
        if np.any(np.abs(cross_products) < 1e-10):
            return 1e6
        if not (np.all(cross_products > 0) or np.all(cross_products < 0)):
            return 1e6
            
        # Check repeated points
        # Distance between consecutive points
        edge_lens = np.linalg.norm(edges, axis=1)
        if np.any(edge_lens < 1e-10):
            return 1e6
            
        # Calculate vertex scores
        min_score = 1.0
        
        for i in range(n):
            # Distances from vertex i to all others
            diffs = polygon_norm - polygon_norm[i]
            d = np.linalg.norm(diffs, axis=1)
            d[i] = np.inf # Exclude self
            
            # Sort the 19 distances
            d_sorted = np.sort(d)
            
            # Check min distance
            if d_sorted[0] < 0.05:
                min_score = 0.0
                break
                
            # Find k: first index minimizing d_{k+1} - d_k
            gaps = d_sorted[1:] - d_sorted[:-1]
            # Handle case where gaps might be negative due to float error? 
            # Sorted array should have non-negative gaps.
            k = np.argmin(gaps)
            
            # D = (d_k + d_{k+1}) / 2
            # Note: k is index in gaps array (0..17). 
            # d_k corresponds to d_sorted[k], d_{k+1} to d_sorted[k+1].
            D = (d_sorted[k] + d_sorted[k+1]) / 2.0
            
            if D < 0.1:
                min_score = 0.0
                break
                
            # Select 4 distances
            if k == 0:
                idxs = [0, 1, 2, 3]
            elif k == 17:
                idxs = [15, 16, 17, 18]
            else:
                idxs = [k-1, k, k+1, k+2]
                
            d_sel = d_sorted[idxs]
            
            # raw = mean of max(d/D, D/d)
            ratios = np.maximum(d_sel / D, D / d_sel)
            raw = np.mean(ratios)
            
            vertex_score = min(1.0, 1.0 / raw)
            
            if vertex_score < min_score:
                min_score = vertex_score
                
        # We want to maximize min_score.
        # Return negative for minimizer.
        # Add small penalty if not strictly convex? Already handled above.
        return -min_score

    # Optimize
    # Bounds: radii must be positive. Let's say between 0.5 and 1.5
    bounds = [(0.5, 1.5)] * n
    
    # Use Nelder-Mead
    try:
        result = minimize(calculate_score, r_init, method='Nelder-Mead', bounds=bounds, 
                          options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
        r_opt = result.x
    except Exception:
        r_opt = r_init
        
    # Final construction
    x = r_opt * np.cos(angles)
    y = r_opt * np.sin(angles)
    polygon = np.stack([x, y], axis=1)
    
    # Normalize
    centroid = np.mean(polygon, axis=0)
    polygon = polygon - centroid
    dists_centroid = np.linalg.norm(polygon, axis=1)
    mean_dist = np.mean(dists_centroid)
    
    if mean_dist <= 0.01:
        # Fallback
        angles_reg = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        polygon = np.stack([np.cos(angles_reg), np.sin(angles_reg)], axis=1)
    else:
        polygon = polygon / mean_dist
        
    return polygon

# EVOLVE-BLOCK-END
