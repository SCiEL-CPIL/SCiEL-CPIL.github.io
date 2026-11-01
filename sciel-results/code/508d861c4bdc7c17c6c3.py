import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    # 1. Generate hexagonal lattice points
    # We need enough points to select N=50 from the center
    # Spacing slightly less than optimal to ensure we have candidates
    # Optimal spacing for N=50 is approx 2*0.1258 = 0.2516
    # Let's use a spacing of 0.24 to be safe and get more points
    spacing = 0.24
    
    # Generate a grid of hex lattice points
    # x = i * spacing + (j % 2) * spacing / 2
    # y = j * spacing * sqrt(3) / 2
    # We need to cover the unit circle
    max_extent = 2.0
    i_range = range(-int(max_extent/spacing)-2, int(max_extent/spacing)+3)
    j_range = range(-int(max_extent/(spacing*np.sqrt(3)/2))-2, int(max_extent/(spacing*np.sqrt(3)/2))+3)
    
    points = []
    for i in i_range:
        for j in j_range:
            x = i * spacing + (j % 2) * spacing * 0.5
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                points.append([x, y])
    
    points = np.array(points)
    
    # 2. Select N points closest to the origin
    # This mimics the seed program's logic but with a specific spacing
    dists = np.linalg.norm(points, axis=1)
    sorted_indices = np.argsort(dists)
    initial_points = points[sorted_indices[:N]].copy()
    
    # 3. Define objective function
    def objective(x):
        pts = x.reshape(-1, 2)
        # Check boundary
        dists_to_boundary = 1.0 - np.linalg.norm(pts, axis=1)
        min_boundary = np.min(dists_to_boundary)
        
        # Check pair distances
        # Vectorized calculation of pairwise distances
        # d_ij = norm(pts[i] - pts[j])
        # We only need the minimum
        # For N=50, 1225 pairs. This is fast enough to compute in a loop or vectorized.
        # Vectorized:
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)
        # Set diagonal to inf to ignore self-distance
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair = np.min(dist_matrix)
        
        # The radius is half the minimum distance
        r = min(min_pair, min_boundary) / 2.0
        
        # We want to maximize r, so minimize -r
        return -r

    # 4. Optimize
    # Use Powell method as it is derivative-free and often works well for this
    # Set a time limit guard
    start_time = time.time()
    max_time = 100.0 # Leave margin for safety
    
    # Initial guess
    x0 = initial_points.flatten()
    
    # Run optimization
    # Powell can be slow, so we might limit iterations
    # Nelder-Mead is another option
    try:
        res = minimize(objective, x0, method='Powell', options={'maxiter': 1000, 'maxfev': 50000})
        
        # Check time
        if time.time() - start_time > max_time:
             # Return best found so far? Powell doesn't return intermediate best easily without callback.
             # For now, if it times out, we return the initial points or the result if it finished.
             # Actually, minimize will stop if maxiter/maxfev is reached.
             pass
    except Exception:
        # If optimization fails, return initial points
        pass

    optimized_points = res.x.reshape(-1, 2)
    
    # 5. Validation and cleanup
    # Ensure points are finite
    if not np.all(np.isfinite(optimized_points)):
        optimized_points = initial_points
        
    # Ensure points are within the unit circle (hard constraint check)
    # If a point is outside, the score will be low/invalid.
    # We should project them back or just return as is if the optimizer kept them inside.
    # The objective function penalizes going outside, so they should be inside.
    # However, due to floating point errors or optimizer quirks, a point might be slightly outside.
    # Let's check and clip if necessary? No, clipping changes the geometry.
    # The evaluator checks `1 - |c_i|`. If |c_i| > 1, this is negative.
    # The score is `min(...)`. If any term is negative, the score is negative or invalid.
    # The contract says "A centre on or outside the container ... make the candidate invalid".
    # So we MUST ensure |c_i| <= 1.
    
    # Let's check distances to boundary
    norms = np.linalg.norm(optimized_points, axis=1)
    if np.any(norms > 1.0):
        # If any point is outside, we have a problem.
        # We can try to pull them back towards the center.
        # Or, simpler: if the optimizer failed to keep them inside,
        # we should probably fall back to a safer initialization or a constrained solver.
        # For now, let's try to repair by moving them radially inward.
        for i in range(N):
            if norms[i] > 1.0:
                optimized_points[i] = optimized_points[i] * (0.99 / norms[i])
        # Re-check
        norms = np.linalg.norm(optimized_points, axis=1)
        if np.any(norms > 1.0):
             # Still outside? Use initial points which were inside.
             optimized_points = initial_points
             
    return optimized_points
