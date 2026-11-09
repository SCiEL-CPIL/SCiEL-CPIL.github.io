"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    # 1. Generate a hexagonal lattice initialization
    # We aim for roughly 200 points. 
    # A hex grid with 14 rows alternating 15 and 14 points gives 203 points.
    # Rows: 0..13
    # Even rows (0,2,4,6,8,10,12): 15 points
    # Odd rows (1,3,5,7,9,11,13): 14 points
    # Total: 7*15 + 7*14 = 105 + 98 = 203
    
    rows = 14
    points = []
    # Spacing in hex grid
    # Let side of hex be s. Distance between centers horizontally is s.
    # Distance between rows is s * sqrt(3)/2.
    # We will generate in a normalized coordinate system and then scale.
    
    s = 1.0
    h = s * np.sqrt(3) / 2
    
    for i in range(rows):
        if i % 2 == 0:
            n_points = 15
            offset = 0.0
        else:
            n_points = 14
            offset = s / 2.0
            
        for j in range(n_points):
            x = j * s + offset
            y = i * h
            points.append([x, y])
            
    points = np.array(points)
    
    # We have 203 points. We need to remove 3.
    # Remove points that are furthest from the center to keep the core dense and symmetric-ish.
    center = points.mean(axis=0)
    dists = np.linalg.norm(points - center, axis=1)
    # Sort by distance, keep the 200 closest to the center
    # This effectively trims the "corners" of the hexagonal block
    sorted_indices = np.argsort(dists)
    points = points[sorted_indices[:N]]
    
    # 2. Scale and translate to fit [-0.5, 0.5]^2
    # Calculate current bounding box
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    
    # We want to fit this into [-0.5, 0.5].
    # The side length of the square is 1.
    # We should scale such that the points fit, but we want to maximize radius, 
    # so we should scale to fill the square as much as possible.
    # However, the aspect ratio of the hex block is not 1:1.
    # We will scale uniformly to fit the largest dimension into 1.
    
    width = max_x - min_x
    height = max_y - min_y
    
    scale = 1.0 / max(width, height)
    
    # Center the points
    points = (points - center) * scale
    # Shift to be centered at 0
    # The points are now centered at 0, but we need to ensure they are within [-0.5, 0.5]
    # Since we scaled by max(width, height), one dimension is 1.0 (from -0.5 to 0.5)
    # The other dimension is smaller, so it will be strictly inside.
    
    # 3. Optimization
    # Objective: Minimize negative of minimum radius
    def objective(x):
        pts = x.reshape(-1, 2)
        # Check bounds
        # min distance to boundary
        dx = 0.5 - np.abs(pts[:, 0])
        dy = 0.5 - np.abs(pts[:, 1])
        min_boundary = np.min(np.minimum(dx, dy))
        
        # min distance between points
        # Calculate pairwise distances
        # Using broadcasting
        # dist[i, j] = norm(pts[i] - pts[j])
        # We only need min for i < j
        # To save memory/time, we can do this in chunks or use scipy.spatial.distance
        from scipy.spatial.distance import pdist
        dists = pdist(pts)
        min_pair = np.min(dists) / 2.0
        
        r = min(min_boundary, min_pair)
        return -r

    # Initial guess
    x0 = points.flatten()
    
    # Options for Nelder-Mead
    # maxiter might need to be high, but we have time limit.
    # tol can be loose.
    res = minimize(objective, x0, method='Nelder-Mead', 
                   options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-6, 'disp': False})
    
    # Get optimized points
    optimal_pts = res.x.reshape(-1, 2)
    
    # Ensure points are strictly inside (clip slightly if needed due to numerical errors)
    # The objective function penalizes going outside, so they should be inside.
    # But just in case, clip to [-0.5, 0.5]
    optimal_pts[:, 0] = np.clip(optimal_pts[:, 0], -0.5, 0.5)
    optimal_pts[:, 1] = np.clip(optimal_pts[:, 1], -0.5, 0.5)
    
    return optimal_pts
# EVOLVE-BLOCK-END
