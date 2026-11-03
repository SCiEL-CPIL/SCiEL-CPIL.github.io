"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 200

# EVOLVE-BLOCK-START
def _compute_score(centers):
    """Compute the maximum possible radius for a given set of centers."""
    # Boundary constraint
    # Square is [-0.5, 0.5]
    # Distance to boundary for center (x, y) is min(0.5 - |x|, 0.5 - |y|)
    # Actually, distance to the closest side.
    # Side 1: x = 0.5 -> dist = 0.5 - x
    # Side 2: x = -0.5 -> dist = x + 0.5
    # Side 3: y = 0.5 -> dist = 0.5 - y
    # Side 4: y = -0.5 -> dist = y + 0.5
    
    # Vectorized distance to boundaries
    d_x = np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5)
    d_y = np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
    boundary_dists = np.minimum(d_x, d_y)
    min_boundary = np.min(boundary_dists)
    
    # Pairwise distances
    # Using scipy for efficient distance calculation if available, 
    # but standard numpy is safer for portability if SciPy isn't guaranteed in the "standard library" sense 
    # although the prompt says SciPy is available.
    from scipy.spatial.distance import pdist
    dists = pdist(centers, 'euclidean')
    min_pair = np.min(dists) if len(dists) > 0 else np.inf
    
    return min(min_boundary, min_pair / 2.0)

def _hex_grid_initialization():
    """
    Generate a hexagonal packing arrangement.
    We want to fit N points. 
    Hexagonal packing has rows with alternating shifts.
    Vertical spacing: dy = sqrt(3)/2 * dx
    Horizontal spacing: dx
    Row shift: dx/2 for odd rows.
    
    We estimate the grid size to cover the square efficiently.
    """
    # Approximate optimal aspect ratio for hex packing in a square is roughly 1:1
    # Let's try to fit a grid of roughly sqrt(N) x sqrt(N)
    # But hex packing is denser.
    # Area per point ~ sqrt(3)/2 * r^2 (where r is radius, d=2r)
    # Actually area per point in hex lattice with spacing d is (sqrt(3)/2) * d^2.
    # Total area = 1. N points.
    # N * (sqrt(3)/2) * (2r)^2 <= 1  =>  r^2 <= 1 / (2*sqrt(3)*N)
    # r_max_theoretical ~ 0.5 / sqrt(N * sqrt(3)/2)
    
    # Let's just pick a reasonable initial spacing and let optimization fix it.
    # Start with a grid that slightly overlaps or is sparse? 
    # Better to start with a valid configuration.
    # Square grid is valid. Hex is valid if scaled correctly.
    
    # Let's create a hex grid that fits inside [-0.5, 0.5]
    # We'll iterate on the number of rows/cols to get close to N.
    
    best_centers = None
    best_score = -np.inf
    
    # Try different aspect ratios for the hex grid to see which fits N best
    for aspect in [0.8, 0.9, 1.0, 1.1, 1.2]:
        # Approximate number of rows
        rows = int(np.sqrt(N * aspect))
        cols = int(np.sqrt(N / aspect))
        
        # Refine rows/cols to get exactly N or close, then trim/pad?
        # The prompt requires exactly N points.
        # We can generate a larger hex grid and take the first N, or center a subset.
        # Taking a subset of a hex grid breaks the regularity at the edges, 
        # but optimization will fix it.
        
        # Let's generate a grid slightly larger than N
        target_rows = int(np.ceil(np.sqrt(N * aspect)))
        target_cols = int(np.ceil(np.sqrt(N / aspect)))
        
        # Ensure we have at least N points
        while target_rows * target_cols < N:
            target_rows += 1
            target_cols += 1
            
        # Generate hex grid
        # Spacing dx, dy = sqrt(3)/2 * dx
        # We want the grid to fit in [-0.5, 0.5]
        # Width = (cols - 1) * dx + dx (if shifted?) 
        # Let's just define dx and dy based on fitting the grid into the square.
        
        # Fit cols in x, rows in y
        # (cols - 1) * dx <= 1.0  => dx <= 1.0 / (cols - 1)
        # (rows - 1) * dy <= 1.0  => dy <= 1.0 / (rows - 1)
        # dy = sqrt(3)/2 * dx
        
        # Constraint 1: dx <= 1.0 / max(1, cols - 1)
        # Constraint 2: sqrt(3)/2 * dx <= 1.0 / max(1, rows - 1) => dx <= 2.0 / (sqrt(3) * max(1, rows - 1))
        
        max_dx_x = 1.0 / max(1, cols - 1) if cols > 1 else np.inf
        max_dx_y = 2.0 / (np.sqrt(3) * max(1, rows - 1)) if rows > 1 else np.inf
        
        dx = min(max_dx_x, max_dx_y)
        dy = np.sqrt(3) / 2.0 * dx
        
        # Center the grid
        x_coords = (np.arange(target_cols) - (target_cols - 1) / 2.0) * dx
        y_coords = (np.arange(target_rows) - (target_rows - 1) / 2.0) * dy
        
        centers_list = []
        for r_idx, y in enumerate(y_coords):
            for c_idx, x in enumerate(x_coords):
                # Shift odd rows by dx/2
                if r_idx % 2 == 1:
                    x += dx / 2.0
                centers_list.append([x, y])
                
        centers = np.array(centers_list)
        
        # We have more than N points. 
        # Strategy: Keep the points closest to the center? 
        # Or just take the first N? 
        # Taking first N biases to top-left. 
        # Better: Sort by distance to origin and take the N closest to origin?
        # This keeps the "core" of the packing.
        
        dists_to_origin = np.sum(centers**2, axis=1)
        indices = np.argsort(dists_to_origin)
        centers = centers[indices[:N]]
        
        score = _compute_score(centers)
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
            
    return best_centers

def _local_optimization(centers, time_limit=100):
    """
    Force-directed relaxation using a spatial hash grid.
    """
    # Convert to list of lists for easier manipulation if needed, 
    # but numpy is faster. We'll stick to numpy.
    
    # Spatial hash grid parameters
    # Cell size should be larger than the current minimum distance to ensure neighbors are found.
    # We'll update cell size dynamically or use a fixed reasonable size.
    # Initial radius is ~0.036. Min dist ~0.07.
    # Let's use cell size = 0.1 initially.
    
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Current radius estimate
    r = _compute_score(centers)
    
    # Precompute pairwise indices? No, N=200 is small enough for O(N^2) in pure Python loops?
    # 200*200 = 40,000 ops per step. 100,000 steps = 4e9 ops. Too slow in Python.
    # We MUST use vectorized numpy or spatial hashing.
    
    # Vectorized approach for N=200:
    # 1. Compute all pairwise distances (200x200 matrix).
    # 2. Find min distance.
    # 3. Compute forces for pairs closer than some cutoff (e.g., 2*r).
    
    # Since N is small (200), a full NxN matrix is 40k floats. 
    # Computing this 10,000 times is 400M float ops. NumPy can do this in ~1-2s per 1000 iterations?
    # Let's test speed mentally. 
    # 200x200 distance matrix calculation is very fast in NumPy.
    
    step_size = 0.001
    cooling = 0.999
    
    # Main loop
    while time.time() < deadline:
        # 1. Compute pairwise distances
        # centers shape (N, 2)
        # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        # dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # This allocates a 200x200x2 array and a 200x200 array.
        # Memory is fine. Speed is fine.
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Mask diagonal
        np.fill_diagonal(dists, np.inf)
        
        # Find min distance
        min_dist = np.min(dists)
        current_r = min_dist / 2.0
        
        # Check boundary
        d_x = np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5)
        d_y = np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
        boundary_dists = np.minimum(d_x, d_y)
        min_boundary = np.min(boundary_dists)
        
        r = min(current_r, min_boundary)
        
        # If we've reached a local optimum where no move improves, we might need to break.
        # But force-directed will naturally push apart.
        
        # Compute forces
        # Force is repulsive if dist < 2*r_target? 
        # Actually, we want to maximize the minimum distance.
        # This is equivalent to minimizing the sum of inverse powers of distances for close pairs?
        # Or simply: push apart any pair that is closer than the current global minimum?
        # No, that's unstable.
        
        # Better: Use a potential energy function.
        # E = sum (1/d_ij^2) for all pairs.
        # Minimize E.
        
        # Gradient of E w.r.t center i:
        # dE/dxi = sum_j -2 * (xi - xj) / d_ij^4
        
        # We only care about pairs with d_ij < cutoff.
        # Cutoff can be 2 * current_min_dist.
        
        cutoff = 2.0 * min_dist
        mask = dists < cutoff
        
        # Zero out diagonal in mask
        np.fill_diagonal(mask, False)
        
        # Compute forces
        # forces = np.zeros_like(centers)
        # For each i, sum over j where mask[i,j] is True:
        # force_i += -2 * (centers[i] - centers[j]) / dists[i,j]^4
        
        # Vectorized:
        # We need to sum (diff / dist^4) * mask
        # diff is (N, N, 2)
        # dists is (N, N)
        
        # inv_dist4 = np.zeros_like(dists)
        # np.fill_diagonal(inv_dist4, 0)
        # inv_dist4[mask] = 1.0 / dists[mask]**4
        
        # force_density = diff * inv_dist4[:, :, np.newaxis]
        # forces = -2.0 * np.sum(force_density, axis=1)
        
        # This involves creating large intermediate arrays.
        # Let's optimize.
        
        # Only compute for masked pairs?
        # Easier: Just use the full matrix, it's small.
        
        # Handle zero distances (coincident points)
        # If dist is 0, force is undefined. Add small jitter.
        # We assume no coincident points from initialization.
        
        # Calculate forces
        # d^4 in denominator
        # To avoid division by zero, we already masked diagonal.
        # But if two distinct points are very close, d^4 is small -> huge force.
        # This is desired to separate them.
        
        # Let's add a small epsilon to dists to avoid inf
        safe_dists = np.where(dists > 1e-9, dists, 1e-9)
        
        # Force factor: 1 / d^4
        # We want to minimize sum 1/d^2.
        # Gradient is -2 * r_vec / d^4.
        
        force_factor = np.zeros_like(dists)
        # Only apply force if within cutoff to save computation? 
        # Actually, 1/d^4 decays fast. 
        # But for N=200, full matrix is fine.
        
        force_factor[mask] = 1.0 / (safe_dists[mask] ** 4)
        
        # forces[i] = -2 * sum_j (centers[i] - centers[j]) * force_factor[i,j]
        # Note: diff[i,j] = centers[i] - centers[j]
        
        # force_contrib = diff * force_factor[:, :, np.newaxis]
        # forces = -2.0 * np.sum(force_contrib, axis=1)
        
        # This creates a (200, 200, 2) array. 80k floats. Fine.
        
        force_contrib = diff * force_factor[:, :, np.newaxis]
        forces = -2.0 * np.sum(force_contrib, axis=1)
        
        # Boundary forces
        # If a point is too close to the boundary, push it inward.
        # Boundary distance d_b.
        # If d_b < r_target (or some threshold), push inward.
        # Let's push inward if d_b < min_dist / 2 (i.e., radius would exceed boundary)
        # Actually, we want to maximize min(min_pair/2, min_boundary).
        # So if boundary is the bottleneck, we should push points away from boundary.
        
        # Simple heuristic: 
        # If min_boundary < min_dist / 2, apply strong inward force to points near boundary.
        
        if min_boundary < current_r:
             # Boundary is the limiting factor
             # Push points with small boundary distance inward
             # Force direction: towards center
             # Magnitude: proportional to how close to boundary
             
             # d_x, d_y computed above
             # If x > 0, force_x is negative (push left). If x < 0, force_x is positive (push right).
             # Actually, push towards 0.
             
             # Force = -k * position (spring to center) for points near boundary?
             # Or simply: if x > 0.5 - r, force_x -= (x - (0.5 - r))
             
             # Let's use a soft constraint:
             # F_boundary_x = -x * k if |x| > 0.5 - r
             # Let's just add a force towards the center for all points, weighted by closeness to boundary.
             
             # Simpler: 
             # If point is outside the "safe zone" (0.5 - r), push it back.
             # Safe zone: |x| < 0.5 - r, |y| < 0.5 - r
             
             # Let's just use the gradient of the boundary term in the objective?
             # Objective: min( min_pair/2, min_boundary )
             # This is non-smooth.
             
             # Heuristic: Add force towards center for points with d_b < r
             mask_boundary = boundary_dists < current_r
             # Force = -centers * strength
             # Strength increases as d_b decreases
             # Let's just push them towards the center with a fixed force
             forces[mask_boundary] -= centers[mask_boundary] * 0.1
             
        # Apply forces
        # Normalize forces to avoid explosion?
        # Or use step_size.
        
        # Step size should be small to avoid oscillation.
        # Adaptive step size:
        # max_force = np.max(np.linalg.norm(forces, axis=1))
        # if max_force > 0:
        #     step = forces / max_force * step_size
        
        # Simple step
        centers += forces * step_size
        
        # Project back to square if they went out?
        # The force should keep them in, but just in case:
        centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
        centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
        
        # Cool down
        step_size *= cooling
        
        # If step_size becomes too small, break
        if step_size < 1e-8:
            break

    return centers

def construct():
    # 1. Generate initial hex grid
    centers = _hex_grid_initialization()
    
    # 2. Optimize
    # Leave 20 seconds for safety
    centers = _local_optimization(centers, time_limit=100)
    
    # 3. Final check and return
    return centers
# EVOLVE-BLOCK-END
