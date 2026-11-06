"""
Packing 200 equal circles in a unit square using direct push-apart local search.
Strategy:
1. Initialize with hexagonal and square grids.
2. Use a "push-apart" algorithm that directly resolves the minimum distance constraint 
   and wall constraints without gradient computation. This is robust to local minima 
   when combined with random restarts.
3. Time-boxed execution to ensure valid return within limits.
"""
import numpy as np
import time

N = 200

def _compute_score(centers):
    """Compute the radius for a given set of centers."""
    if len(centers) == 0:
        return 0.0
    # Distance to walls
    wall_dist = np.min(np.abs(0.5 - np.abs(centers)))
    # Distance to neighbors
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = np.min(dist_sq)
    min_dist = np.sqrt(min_dist_sq)
    r = min(wall_dist, min_dist / 2.0)
    return r

def _generate_hex_grid(N, d_scale=1.0):
    """Generate a hexagonal grid of points."""
    d = np.sqrt(2.0 / (N * np.sqrt(3.0))) * d_scale
    points = []
    row_spacing = d * np.sqrt(3.0) / 2.0
    num_rows = int(np.ceil(1.0 / row_spacing)) + 2
    
    for row in range(num_rows):
        y = -0.5 + row * row_spacing
        if y > 0.5:
            break
        offset = d / 2.0 if row % 2 else 0.0
        num_cols = int(np.ceil(1.0 / d)) + 2
        for col in range(num_cols):
            x = -0.5 + col * d + offset
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
                
    points = np.array(points)
    if len(points) < N:
        return None
    return points[:N]

def _generate_square_grid(N):
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]

def _push_apart(centers, max_iter=5000, tol=1e-10):
    """
    Iteratively push apart overlapping circles and pull in circles outside bounds.
    This is a direct constraint satisfaction method.
    """
    centers = centers.copy()
    n = len(centers)
    
    for _ in range(max_iter):
        # 1. Check wall constraints
        # For each point, if it's too close to a wall, push it in.
        # We can do this vectorized, but we must be careful not to cause overlaps.
        # A simple approach: iterate points, fix walls, then fix overlaps.
        
        # Wall fixing
        # x boundaries: [-0.5, 0.5]
        # If x < -0.5, x = -0.5. If x > 0.5, x = 0.5.
        # This is projection onto the box.
        centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
        centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
        
        # 2. Check pairwise distances
        # Find the pair with minimum distance.
        # If min_dist < 2*r (where r is current min radius?), we need to push them apart.
        # Actually, we want to maximize the minimum distance.
        # So, find the current minimum distance d_min.
        # If d_min is less than some target (or just less than the current best), 
        # we push the closest pair apart.
        
        # Efficiently finding closest pair in 2D for N=200:
        # O(N^2) is 40000 ops, which is fast enough.
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        
        min_idx = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
        i, j = min_idx
        d = np.sqrt(dist_sq[i, j])
        
        # The "radius" we are effectively trying to achieve is d/2.
        # We want to increase d.
        # Push i and j apart by a small amount.
        
        if d < 1e-10:
            # Coincident points, nudge randomly
            centers[i] += np.random.normal(0, 0.01, 2)
            centers[j] -= np.random.normal(0, 0.01, 2)
        else:
            v = (centers[i] - centers[j]) / d
            # Push each away from the other by half the "excess" or a fixed step?
            # To maximize the minimum distance, we should push them until they are 
            # no longer the unique minimum, or until they hit a wall/another circle.
            # A simple gradient step: move i by +v*step, j by -v*step.
            # What step size? 
            # If we move them by d/2, they will be at distance d+d. 
            # But this might violate other constraints.
            # Let's use a small step size to allow convergence.
            step = 0.01
            centers[i] += v * step
            centers[j] -= v * step
            
        # Re-clip to box after movement
        centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
        centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
        
        # Convergence check: if the min distance hasn't changed significantly, stop?
        # Or just run for max_iter.
        
    return centers

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    best_centers = None
    best_score = -1.0
    
    # Initializations
    inits = []
    
    hex_grid = _generate_hex_grid(N)
    if hex_grid is not None:
        inits.append(hex_grid)
        
    square_grid = _generate_square_grid(N)
    inits.append(square_grid)
    
    # Random jittered grids
    np.random.seed(42)
    for _ in range(5):
        jittered = square_grid.copy()
        jittered += np.random.normal(0, 0.05, jittered.shape)
        inits.append(jittered)
        
    # Pure random
    for _ in range(5):
        rand_centers = np.random.uniform(-0.49, 0.49, (N, 2))
        inits.append(rand_centers)

    for init_centers in inits:
        if time.time() - start_time > time_limit:
            break
            
        # Run push-apart optimization
        # We can run it for a fixed number of iterations or until time runs out.
        # Let's run for a few thousand iterations.
        optimized = _push_apart(init_centers, max_iter=2000)
        
        score = _compute_score(optimized)
        
        if score > best_score:
            best_score = score
            best_centers = optimized
            
    # If no optimization succeeded (shouldn't happen), return best init
    if best_centers is None:
        best_centers = inits[0]
        
    return best_centers
