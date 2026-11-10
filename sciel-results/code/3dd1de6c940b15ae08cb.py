# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def equidistant_points_polygon20() -> np.ndarray:
    """
    Builds a convex 20-gon optimized to maximize the minimum vertex score.
    The score is defined by the clustering of the 4 nearest neighbors of each vertex.
    We optimize the radial distances of the vertices (assuming fixed angular positions)
    to make the 4 nearest neighbors of each vertex as equidistant as possible.
    """
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    def radii_to_points(radii):
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.stack([x, y], axis=1)

    def get_raw_scores(points):
        raw_scores = []
        for i in range(n):
            v = points[i]
            # Calculate distances to all other vertices
            dists = np.linalg.norm(points - v, axis=1)
            # Exclude self-distance (which is 0)
            # The sorted list will have 0 at index 0.
            # We need distances to the *other* vertices.
            # So we sort the non-zero distances.
            # Since v is at points[i], dists[i] is 0.
            # We can just remove it or mask it.
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            d = dists[mask]
            d = np.sort(d)
            
            # d has length 19
            # Find k: first index minimizing d[k+1] - d[k]
            gaps = d[1:] - d[:-1]
            k = np.argmin(gaps)
            
            # Calculate D
            D = (d[k] + d[k+1]) / 2.0
            
            # Check D < 0.1 condition (though with normalized polygon, D should be > 0.1 usually)
            # The problem statement says "if D < 0.1 the vertex scores 0".
            # Our polygon will be normalized, so distances are roughly 1.
            
            # Select 4 distances
            if k == 0:
                selected = d[0:4]
            elif k == 17: # 19-2 = 17
                selected = d[15:19]
            else:
                selected = d[k-1:k+3]
            
            # Calculate raw score
            # raw = mean of max(d/D, D/d)
            ratios = np.maximum(selected / D, D / selected)
            raw = np.mean(ratios)
            raw_scores.append(raw)
            
        return raw_scores

    def loss(radii):
        # Ensure radii are positive
        if np.any(radii <= 0.1):
            return 1e6
            
        points = radii_to_points(radii)
        
        # Check strict convexity
        # Cross product of consecutive edges
        # Edge i: points[(i+1)%n] - points[i]
        edges = np.roll(points, -1, axis=0) - points
        cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
        if np.any(cross <= 0):
            return 1e6 + np.sum(cross[cross <= 0])
            
        raw_scores = get_raw_scores(points)
        
        # We want to minimize the maximum raw score, or sum of squares of excess
        # Target raw score is 1.0
        excess = np.maximum(0, np.array(raw_scores) - 1.0)
        return np.sum(excess**2)

    # Initial guess: regular polygon
    initial_radii = np.ones(n)
    
    # Bounds to maintain convexity and avoid degenerate shapes
    bounds = [(0.5, 2.0) for _ in range(n)]
    
    # Optimize
    # Use Nelder-Mead for simplicity and robustness to non-differentiable points (like argmin)
    # L-BFGS-B might struggle with the discrete nature of 'k'
    res = minimize(loss, initial_radii, method='Nelder-Mead', 
                   options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
    
    optimal_radii = res.x
    
    # Fallback to regular if optimization fails or results in non-convex
    if res.fun > 1e5:
        optimal_radii = initial_radii
        
    points = radii_to_points(optimal_radii)
    
    # Normalize as per task contract
    # 1. Subtract centroid
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    # 2. Divide by mean distance to centroid
    dists_to_centroid = np.linalg.norm(points, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    
    if mean_dist <= 0.01:
        # Fallback
        return radii_to_points(initial_radii)
        
    points = points / mean_dist
    
    return points

# EVOLVE-BLOCK-END
