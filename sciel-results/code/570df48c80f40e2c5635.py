import numpy as np
from scipy.optimize import minimize
import time

N = 200

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin

    # 1. Initialization: Hexagonal Grid
    # Approximate optimal spacing s ~ 0.075
    s = 0.075
    # Generate a grid larger than needed
    # x range: -0.5 to 0.5
    # y range: -0.5 to 0.5
    # Hex grid:
    # x_i = i*s + (j%2)*s/2
    # y_j = j * sqrt(3)/2 * s
    
    # We want to center this grid.
    # Let's generate points and then filter/center.
    
    points = []
    # Estimate number of rows/cols needed
    # Width = 1.0, Height = 1.0
    # cols ~ 1.0/s = 13.3 -> 15
    # rows ~ 1.0/(s*sqrt(3)/2) = 1.0/0.065 = 15.3 -> 17
    max_idx = 20
    
    for j in range(-max_idx, max_idx + 1):
        for i in range(-max_idx, max_idx + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * np.sqrt(3) / 2.0 * s
            points.append((x, y))
            
    points = np.array(points)
    
    # Center the grid roughly
    # Find the centroid of the points that fall inside a slightly larger box
    # Or just shift the whole grid so the "center" is at 0,0
    # A simpler way: Generate points in [0, W] x [0, H] and shift.
    # Let's regenerate to ensure good coverage.
    
    # Better initialization:
    # Create a grid of points in [0,1]x[0,1] using hex packing
    # Then scale and shift to [-0.5, 0.5]
    
    s_init = 1.0 / 15.0 # Initial guess
    hex_points = []
    for j in range(20):
        for i in range(20):
            x = i * s_init + (j % 2) * s_init / 2.0
            y = j * np.sqrt(3) / 2.0 * s_init
            hex_points.append([x, y])
    hex_points = np.array(hex_points)
    
    # Filter points that are roughly within the square [0,1]x[0,1]
    # This filtering is tricky because the hex grid is skewed.
    # Let's just take the first 200 points from a dense hex grid centered at 0,0
    # and scale them to fit.
    
    # Re-do initialization more robustly
    # We want 200 points.
    # Let's use a square grid first as it's easier to center, then jitter.
    # Square grid 15x15 = 225 points.
    # Remove 25 points (e.g., corners or random) to get 200.
    # This is a valid start.
    
    m = 15
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    # Grid is 225 points.
    # We need 200.
    # Let's just use the first 200? That leaves a gap at the end.
    # Better: Use a 14x14 grid (196) + 4 points?
    # Or just use 15x15 and let the optimizer move the extra points?
    # No, we must return exactly N points.
    
    # Let's use a 14x15 grid? 210 points.
    # Let's stick to the 15x15 grid and remove 25 points randomly or from the boundary.
    # Actually, a 15x15 grid has spacing 1/15.
    # If we remove points, the spacing increases locally.
    
    # Let's try a different init:
    # Random points inside the square.
    # np.random.seed(42)
    # x0 = np.random.rand(N, 2) - 0.5
    
    # Random init is bad for local optimization.
    # Grid init is better.
    
    # Let's use the 15x15 grid, and select 200 points that are "most central" or just take a subset.
    # Taking the first 200 from the flattened 15x15 grid (row-major) means we drop the last 25 points (bottom right corner).
    # This creates an asymmetry.
    # Let's drop points from the corners to maintain symmetry.
    
    # 15x15 grid indices 0..14.
    # Drop corners: (0,0), (14,0), (0,14), (14,14) -> 4 points.
    # Drop next layer corners?
    # We need to drop 25 points.
    # Let's just use the grid and let the optimizer fix it.
    # We will use all 225 points? No, N=200.
    
    # Strategy:
    # 1. Create 15x15 grid.
    # 2. Identify 25 points to remove.
    #    Remove points closest to the 4 corners?
    #    Or just take a random subset?
    #    Let's take the 200 points closest to the center (0,0).
    
    dist_to_center = np.sum(grid**2, axis=1)
    idx_sorted = np.argsort(dist_to_center)
    x0 = grid[idx_sorted[:N]]
    
    # 2. Optimization
    # Objective: Minimize -min_dist
    # This is non-smooth. Nelder-Mead is suitable.
    
    def neg_min_dist(x):
        pts = x.reshape(-1, 2)
        # Check bounds
        # If out of bounds, return a large penalty
        # min dist to wall
        wall_dists = np.min(np.array([0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1])]), axis=1)
        min_wall = np.min(wall_dists)
        if min_wall < 0:
            return 1e6 - min_wall # Penalty for being outside
            
        # Pairwise distances
        # Use cdist or manual calculation
        # Manual is faster for small N
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Set diagonal to inf
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        return -min(min_wall, min_pair)
        
    # Nelder-Mead options
    # maxiter needs to be high enough to converge, but time-limited.
    # Each function call is ~20k ops. Very fast.
    # We can do thousands of iterations.
    
    best_x = x0
    best_f = neg_min_dist(x0)
    
    # Run optimization
    # Use multiple restarts?
    # With 110s, we can afford several runs.
    
    for restart in range(5):
        if time.time() > deadline - 10:
            break
            
        if restart == 0:
            x_start = x0
        else:
            # Jitter the previous best
            x_start = best_x + np.random.normal(0, 0.01, size=(N, 2))
            # Clip to valid range
            x_start = np.clip(x_start, -0.49, 0.49)
            
        res = minimize(neg_min_dist, x_start.flatten(), method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
        
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x.reshape(N, 2)
            
        if time.time() > deadline - 5:
            break
            
    # Final check and clip
    best_x = np.clip(best_x, -0.5, 0.5)
    return best_x
