# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a multi-start optimization approach with SLSQP to maximize the sum of log distances,
    which correlates with maximizing the min distance while keeping the scale fixed.
    """
    n = 16
    d = 2
    
    # Objective function to minimize (negative sum of log distances)
    def objective(params):
        pts = params.reshape(n, d)
        # Calculate pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        
        # Avoid log(0)
        dists = np.maximum(dists, 1e-9)
        
        # Sum over upper triangle
        iu = np.triu_indices(n, k=1)
        sum_log_dist = np.sum(np.log(dists[iu]))
        
        return -sum_log_dist

    # Constraints functions
    def centroid_x(params):
        pts = params.reshape(n, d)
        return np.sum(pts[:, 0])
        
    def centroid_y(params):
        pts = params.reshape(n, d)
        return np.sum(pts[:, 1])
        
    def norm_sq(params):
        pts = params.reshape(n, d)
        return np.sum(pts**2) - n

    constraints = [
        {'type': 'eq', 'fun': centroid_x},
        {'type': 'eq', 'fun': centroid_y},
        {'type': 'eq', 'fun': norm_sq}
    ]
    
    # Generate multiple initial guesses
    initial_guesses = []
    
    # 1. 4x4 Grid
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    grid_pts = np.column_stack((xx.ravel(), yy.ravel()))
    grid_pts -= np.mean(grid_pts, axis=0)
    scale = np.sqrt(np.sum(grid_pts**2) / n)
    if scale > 0:
        grid_pts /= scale
    initial_guesses.append(grid_pts)
    
    # 2. Hexagonal Lattice (approximate)
    # Generate a hexagonal grid of 16 points
    # Side length 1 hexagon has 7 points. We need 16.
    # Let's just generate a dense hex grid and pick the best 16? 
    # Or simply create a hex pattern:
    # Row 1: 4 points, Row 2: 4 points, Row 3: 4 points, Row 4: 4 points? No, that's grid.
    # Hexagonal packing:
    # y = 0: x = -1.5, -0.5, 0.5, 1.5 (4 pts)
    # y = h: x = -1.0, 0.0, 1.0 (3 pts)
    # y = 2h: x = -1.5, -0.5, 0.5, 1.5 (4 pts)
    # y = 3h: x = -1.0, 0.0, 1.0 (3 pts)
    # y = 4h: x = -0.5, 0.5 (2 pts)
    # Total: 4+3+4+3+2 = 16.
    h = np.sqrt(3) / 2
    hex_pts_list = []
    # Row 0
    for x in [-1.5, -0.5, 0.5, 1.5]: hex_pts_list.append([x, 0])
    # Row 1
    for x in [-1.0, 0.0, 1.0]: hex_pts_list.append([x, h])
    # Row 2
    for x in [-1.5, -0.5, 0.5, 1.5]: hex_pts_list.append([x, 2*h])
    # Row 3
    for x in [-1.0, 0.0, 1.0]: hex_pts_list.append([x, 3*h])
    # Row 4
    for x in [-0.5, 0.5]: hex_pts_list.append([x, 4*h])
    
    hex_pts = np.array(hex_pts_list)
    hex_pts -= np.mean(hex_pts, axis=0)
    scale = np.sqrt(np.sum(hex_pts**2) / n)
    if scale > 0:
        hex_pts /= scale
    initial_guesses.append(hex_pts)
    
    # 3. Random points (normalized)
    np.random.seed(42)
    for _ in range(3):
        rand_pts = np.random.randn(n, d)
        rand_pts -= np.mean(rand_pts, axis=0)
        scale = np.sqrt(np.sum(rand_pts**2) / n)
        if scale > 0:
            rand_pts /= scale
        initial_guesses.append(rand_pts)

    best_points = None
    best_score = -np.inf
    
    # Run optimization for each initial guess
    for x0_pts in initial_guesses:
        x0 = x0_pts.ravel()
        try:
            res = minimize(objective, x0, method='SLSQP', constraints=constraints, 
                           options={'maxiter': 2000, 'ftol': 1e-12})
            if res.success or res.fun < best_score:
                # Calculate actual score R for this result
                pts = res.x.reshape(n, d)
                # Re-normalize to be safe? The constraints should handle it.
                diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                dists = np.linalg.norm(diff, axis=-1)
                iu = np.triu_indices(n, k=1)
                d_min = np.min(dists[iu])
                d_max = np.max(dists[iu])
                
                if d_max > 0:
                    R = (d_min / d_max) ** 2
                else:
                    R = 0
                
                # The objective was -sum(log(d)), so lower res.fun is better.
                # But we want to compare R.
                # Since the problem is scale invariant, and we fixed the norm, 
                # maximizing sum(log(d)) is a proxy for maximizing d_min.
                # However, different local minima might have different R.
                # Let's track the actual R.
                
                if R > best_score:
                    best_score = R
                    best_points = pts
                    
        except Exception as e:
            pass
            
    # Fallback if optimization failed completely
    if best_points is None:
        best_points = initial_guesses[0]
        
    # Final check: ensure it's finite and shape is correct
    if not np.all(np.isfinite(best_points)):
        best_points = np.random.randn(n, d)
        
    return best_points

# EVOLVE-BLOCK-END
