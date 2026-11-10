# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution

def equidistant_points_polygon20() -> np.ndarray:
    """
    Uses Differential Evolution to maximize the minimum vertex score for a convex 20-gon.
    """
    n = 20
    
    def compute_metric(points):
        """
        Computes the metric (minimum vertex score) for a given set of points.
        Returns 0 if invalid.
        """
        # Check for repeated points
        d_mat = np.linalg.norm(points[:, np.newaxis] - points[np.newaxis, :], axis=2)
        np.fill_diagonal(d_mat, 1.0)
        if np.any(d_mat < 1e-6):
            return 0.0

        # Check strict convexity
        edges = np.roll(points, -1, axis=0) - points
        cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
        
        # All cross products must be non-zero and same sign
        if np.any(np.abs(cross) < 1e-9):
            return 0.0
        if not (np.all(cross > 0) or np.all(cross < 0)):
            return 0.0

        # Normalize
        centroid = np.mean(points, axis=0)
        points_norm = points - centroid
        dists_to_centroid = np.linalg.norm(points_norm, axis=1)
        mean_dist = np.mean(dists_to_centroid)
        
        if mean_dist <= 0.01:
            return 0.0
            
        points_norm = points_norm / mean_dist
        
        # Calculate vertex scores
        min_score = 1.0
        for i in range(n):
            v = points_norm[i]
            d_all = np.linalg.norm(points_norm - v, axis=1)
            d_all[i] = np.inf 
            
            d_sorted = np.sort(d_all)
            
            if d_sorted[0] < 0.05:
                return 0.0
            
            gaps = d_sorted[1:] - d_sorted[:-1]
            k = np.argmin(gaps)
            
            D = (d_sorted[k] + d_sorted[k+1]) / 2.0
            
            if D < 0.1:
                return 0.0
            
            if k == 0:
                selected = d_sorted[0:4]
            elif k == 17:
                selected = d_sorted[15:19]
            else:
                selected = d_sorted[k-1:k+3]
                
            ratios = np.maximum(selected / D, D / selected)
            raw = np.mean(ratios)
            
            score = min(1.0, 1.0 / raw)
            
            if score < min_score:
                min_score = score
                
        return min_score

    def loss(coords):
        points = coords.reshape(n, 2)
        
        # Convexity check for early exit
        edges = np.roll(points, -1, axis=0) - points
        cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
        
        if np.any(np.abs(cross) < 1e-6):
            return 1e6
        if not (np.all(cross > 0) or np.all(cross < 0)):
            return 1e6

        # Repeated points check
        d_mat = np.linalg.norm(points[:, np.newaxis] - points[np.newaxis, :], axis=2)
        np.fill_diagonal(d_mat, 1.0)
        if np.any(d_mat < 1e-6):
            return 1e6

        metric = compute_metric(points)
        if metric == 0.0:
            return 10.0
        return 1.0 - metric

    # Define bounds for DE
    # We use a box of [-5, 5] to allow for various shapes, 
    # normalization will handle the scale.
    bounds = [(-5.0, 5.0)] * (2 * n)
    
    # Initial population: mix of regular polygon and random convex polygons
    # DE generates its own population, but we can seed it?
    # scipy differential_evolution does not accept initial population directly in all versions, 
    # but we can just run it.
    
    # To ensure we start from a good place, we can run DE with a large popsize.
    # However, DE is stochastic.
    
    # Let's try running DE.
    # maxiter=100, popsize=15
    # tol=1e-4
    
    result = differential_evolution(loss, bounds, maxiter=100, popsize=20, seed=42, tol=1e-4, polish=False)
    
    optimal_coords = result.x
    points = optimal_coords.reshape(n, 2)
    
    # Verify and normalize
    metric = compute_metric(points)
    
    # Fallback to regular polygon if optimization failed or score is low
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    init_points = np.stack([x, y], axis=1)
    
    init_metric = compute_metric(init_points)
    
    if metric < init_metric:
        points = init_points
        metric = init_metric
        
    # Final normalization
    centroid = np.mean(points, axis=0)
    points = points - centroid
    dists_to_centroid = np.linalg.norm(points, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    
    if mean_dist <= 0.01:
        return init_points
        
    points = points / mean_dist
    
    return points

# EVOLVE-BLOCK-END
