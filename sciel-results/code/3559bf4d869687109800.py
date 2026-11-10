# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize_scalar

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a 4x4 square lattice base with a radial power-law distortion.
    """
    n = 16
    
    # Base 4x4 grid centered at origin
    # Coordinates: -1.5, -0.5, 0.5, 1.5
    xs = np.array([-1.5, -0.5, 0.5, 1.5])
    ys = np.array([-1.5, -0.5, 0.5, 1.5])
    xx, yy = np.meshgrid(xs, ys, indexing='xy')
    base_points = np.vstack((xx.ravel(), yy.ravel())).T
    
    def score_from_alpha(alpha):
        # Apply radial warp: r_new = r_old ** alpha
        # Note: This scales distances. The ratio d_min/d_max is invariant to global scaling,
        # so we don't need to normalize the radius, but we must ensure points don't collapse to 0.
        # For alpha < 1, outer points move in, inner move out.
        # For alpha > 1, outer points move out, inner move in.
        
        r = np.linalg.norm(base_points, axis=1)
        # Avoid division by zero or issues at origin if any point is at origin (none are in 4x4 grid)
        # But if alpha is very small, points near center expand a lot?
        # r=0 is not in the grid. Min r is sqrt(0.5^2+0.5^2) = 0.707.
        
        # We want to map r -> r^alpha.
        # To keep the scale reasonable for numerical stability, we can just compute distances directly.
        angles = np.arctan2(base_points[:, 1], base_points[:, 0])
        new_r = r ** alpha
        new_points = np.column_stack((new_r * np.cos(angles), new_r * np.sin(angles)))
        
        dists = pdist(new_points)
        if len(dists) == 0:
            return -np.inf
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max == 0:
            return -np.inf
            
        R = (d_min / d_max) ** 2
        # We want to maximize R, so minimize -R
        return -R

    # Optimize alpha
    # Initial guess: 0.5 or 0.8
    # Search range: 0.1 to 1.5
    result = minimize_scalar(score_from_alpha, bounds=(0.1, 1.5), method='bounded')
    best_alpha = result.x
    
    # Construct final points
    r = np.linalg.norm(base_points, axis=1)
    angles = np.arctan2(base_points[:, 1], base_points[:, 0])
    new_r = r ** best_alpha
    final_points = np.column_stack((new_r * np.cos(angles), new_r * np.sin(angles)))
    
    return final_points

# EVOLVE-BLOCK-END
