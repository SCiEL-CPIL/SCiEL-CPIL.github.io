"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Strategy: 
    # 1. Initialize with a hexagonal lattice (dense, valid).
    # 2. Use Soft-Min (Log-Sum-Exp) objective to maximize the minimum distance.
    #    Objective: maximize min( min_{i<j} |c_i - c_j| / 2, min_i (1 - |c_i|) )
    #    Let r be the target radius. We want to maximize r.
    #    Equivalent to minimizing -r.
    #    We can formulate this as minimizing a penalty function that grows as distances shrink below 2r.
    #    Or, simpler: use a gradient ascent on r by adjusting positions to increase the minimum distance.
    #    
    #    Let's use a direct optimization approach:
    #    Variables: x_1...x_300, y_1...y_300.
    #    Objective: maximize r = min( min_{i<j} d_{ij}/2, min_i (1 - d_i) )
    #    
    #    To make this smooth, we can use a "soft minimum" function.
    #    soft_min(A) = -1/alpha * log(sum(exp(-alpha * a_i)))
    #    As alpha -> infinity, soft_min(A) -> min(A).
    #    
    #    We want to maximize soft_min( [d_{ij}/2 for i<j, 1-d_i for i] ).
    #    This is equivalent to minimizing -soft_min(...).
    #    
    #    However, the number of terms is ~ N^2/2 = 45,000. This might be slow for L-BFGS-B.
    #    
    #    Alternative: Penalty method.
    #    Start with an initial r (from hex lattice).
    #    Try to increase r.
    #    For a fixed r, the constraints are:
    #    d_{ij} >= 2r  for all i<j
    #    d_i <= 1-r    for all i
    #    
    #    We can minimize the violation:
    #    V = sum_{i<j} max(0, 2r - d_{ij})^2 + sum_i max(0, d_i - (1-r))^2
    #    If V == 0, the configuration is valid for radius r.
    #    
    #    Strategy:
    #    1. Generate hex lattice.
    #    2. Compute current r.
    #    3. Increase r slightly (e.g., by 0.001).
    #    4. Minimize V using L-BFGS-B.
    #    5. If V is small enough (valid), accept new r and repeat.
    #    6. If V is not small, backtrack r or try harder optimization.
    
    def hex_lattice_init():
        # Generate a hexagonal lattice of points within the unit disk.
        # We need N points.
        # Start with a large grid and trim.
        # Spacing s. Area per point ~ s^2 * sqrt(3)/2.
        # Total area ~ pi. N ~ pi / (s^2 * sqrt(3)/2) => s ~ sqrt(2*pi / (N*sqrt(3)))
        s_est = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
        
        # Try a few spacings to get at least N points.
        for factor in np.linspace(1.2, 0.8, 20):
            s = s_est * factor
            # Hex lattice generation
            # Basis vectors: (s, 0), (s/2, s*sqrt(3)/2)
            # We'll generate points in a bounding box and filter.
            # Bounding box size: roughly 2/s in each direction.
            # Let's use a simpler approach: generate rows.
            
            points = []
            # Determine range of rows
            # Max y is ~ 1. Row height is s*sqrt(3)/2.
            n_rows = int(2 / (s * np.sqrt(3) / 2)) + 2
            for j in range(-n_rows, n_rows + 1):
                y = j * s * np.sqrt(3) / 2
                # x offset for even/odd rows
                x_offset = 0 if j % 2 == 0 else s / 2
                n_cols = int(2 / s) + 2
                for i in range(-n_cols, n_cols + 1):
                    x = i * s + x_offset
                    if x*x + y*y <= 1.0 - s/2: # Ensure circle fits
                        points.append((x, y))
                        
            if len(points) >= N:
                # Select N points closest to center to maximize density in center?
                # Or just take the first N?
                # Taking closest to center is better for packing.
                points_arr = np.array(points)
                radii = np.hypot(points_arr[:, 0], points_arr[:, 1])
                indices = np.argsort(radii)[:N]
                return points_arr[indices]
                
        # Fallback: random
        return np.random.uniform(-1, 1, (N, 2)) * 0.5

    def compute_r(points):
        # Compute the maximum radius r for given points.
        # r = min( min_{i<j} |c_i - c_j| / 2, min_i (1 - |c_i|) )
        
        # Boundary constraint
        dists_to_center = np.hypot(points[:, 0], points[:, 1])
        r_boundary = np.min(1.0 - dists_to_center)
        
        # Pairwise distances
        # This is O(N^2). For N=300, 90k pairs. Fast enough.
        # Use broadcasting? 300x300 matrix is small.
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        # Set diagonal to inf
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        r_pair = min_pair_dist / 2.0
        
        return min(r_boundary, r_pair)

    def penalty_objective(points_flat, r_target):
        # points_flat is shape (2N,)
        points = points_flat.reshape(N, 2)
        
        # Boundary violation
        dists_to_center = np.hypot(points[:, 0], points[:, 1])
        boundary_viol = np.maximum(0, dists_to_center - (1.0 - r_target))
        pen_boundary = np.sum(boundary_viol**2)
        
        # Pairwise violation
        # To save memory/time, we can do this in chunks or just compute all.
        # 300x300 is fine.
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, 1e10)
        pair_viol = np.maximum(0, 2.0 * r_target - dists)
        pen_pair = np.sum(pair_viol**2)
        
        return pen_boundary + pen_pair

    # 1. Initialize
    points = hex_lattice_init()
    current_r = compute_r(points)
    
    # Store best result
    best_points = points.copy()
    best_r = current_r
    
    # 2. Iterative improvement
    # We will try to increase r in small steps.
    # Step size for r: start small, decrease if failing.
    r_step = 0.001
    
    # Optimization settings
    max_iter_opt = 100
    
    for iter_step in range(100):
        if time.time() > deadline:
            break
        
        # Try to increase r
        target_r = current_r + r_step
        
        # Define bounds for points: x, y in [-1, 1]
        bounds = [(-1.0, 1.0)] * (2 * N)
        
        # Initial point
        x0 = best_points.flatten()
        
        # Optimize
        res = minimize(
            penalty_objective, 
            x0, 
            args=(target_r,), 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': max_iter_opt, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        new_points = res.x.reshape(N, 2)
        
        # Check validity of new points for target_r
        # We need to be careful with floating point.
        # Let's compute the actual r achieved.
        new_r = compute_r(new_points)
        
        if new_r >= target_r - 1e-9:
            # Success! We can use this new configuration.
            # Actually, we might have overshot. Let's use the actual new_r.
            # But we want to ensure we are on a valid path.
            # If new_r > target_r, we are good.
            # Update best.
            if new_r > best_r:
                best_r = new_r
                best_points = new_points.copy()
            
            # Accept this step and try to increase r further.
            current_r = new_r
            # Maybe increase step size slightly if we succeeded easily?
            # Keep it simple: constant step or slight increase.
            r_step = min(r_step * 1.1, 0.005)
            
        else:
            # Failed to reach target_r.
            # Try to find the max r we *could* reach with the current points?
            # Or just decrease step size.
            # Let's check if the optimized points are better than current best.
            if new_r > best_r:
                best_r = new_r
                best_points = new_points.copy()
                current_r = new_r
                
            # Decrease step size
            r_step *= 0.5
            
            # If step size is very small, we might be stuck.
            if r_step < 1e-5:
                # Try a random restart or perturbation?
                # Or just break.
                # Let's try a few more iterations with smaller steps.
                pass

    return best_points
# EVOLVE-BLOCK-END
