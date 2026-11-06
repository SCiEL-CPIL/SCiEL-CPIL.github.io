import numpy as np
import time
from scipy.optimize import minimize

N = 50

def calculate_radius(centers):
    """Calculate the maximum radius for the given centers."""
    # Distance to boundary
    # Centers are in [-0.5, 0.5]
    # Distance to wall is min(0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y)
    # Since centered at 0, 0.5 - |x| and 0.5 - |y|
    d_wall = np.minimum(0.5 - np.abs(centers[:, 0]), 0.5 - np.abs(centers[:, 1]))
    min_wall_dist = np.min(d_wall)
    
    # Distance between centers
    # Use broadcasting to compute all pairwise distances
    # diff[i, j] = centers[i] - centers[j]
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Set diagonal to infinity so it doesn't affect min
    np.fill_diagonal(dists, np.inf)
    min_center_dist = np.min(dists)
    
    # Radius is half the min center distance, limited by wall distance
    r = min(min_center_dist / 2.0, min_wall_dist)
    return r

def objective(centers_flat):
    """Objective function to minimize. We minimize a potential energy that penalizes close approaches."""
    centers = centers_flat.reshape(N, 2)
    
    # Calculate distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, 1.0) # Avoid division by zero
    
    # Pairwise repulsion
    # Use inverse square law for strong repulsion
    # We need to handle the case where distance is 0, but diagonal is set to 1.
    # For off-diagonal, if dist is very small, energy is huge.
    pair_energy = np.sum(1.0 / dists_sq)
    
    # Wall repulsion
    # Distance to walls: 0.5 - |x|, 0.5 - |y|
    # If center is outside, this becomes negative. We want to penalize that heavily.
    # Let's use a smooth penalty for going outside or getting close.
    # d_x = 0.5 - abs(x), d_y = 0.5 - abs(y)
    dx = 0.5 - np.abs(centers[:, 0])
    dy = 0.5 - np.abs(centers[:, 1])
    
    # If dx or dy is negative, the circle is outside.
    # We can use (1/d^2) but if d<=0, this blows up or becomes invalid.
    # Better: use a large penalty if d < epsilon.
    # Or simply maximize the minimum distance directly using a soft-min in the objective.
    
    # Let's stick to the potential energy approach but ensure stability.
    # Add a small epsilon to avoid singularity
    eps = 1e-6
    wall_dist_x = dx + eps
    wall_dist_y = dy + eps
    
    # If a point is outside, dx < 0. We want to push it back.
    # Using 1/d^2 works well for d>0. For d<0, it's undefined or negative.
    # Let's use a penalty function: P(d) = 1/d^2 if d > 0, else (1000 + 1000*(0-d))^2
    # Simpler: just use the raw distance squared in the denominator, but clip to a small positive value.
    # This effectively creates a barrier.
    
    wall_energy_x = np.sum(1.0 / (wall_dist_x**2))
    wall_energy_y = np.sum(1.0 / (wall_dist_y**2))
    
    total_energy = pair_energy + wall_energy_x + wall_energy_y
    
    return total_energy

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # Hexagonal lattice initialization
    # Side length 1.
    # Approximate radius for 50 circles in hex packing:
    # Area per circle ~ 1/sqrt(3) * (2r)^2 ? No.
    # Hex lattice density is pi/sqrt(12) ~ 0.9069.
    # 50 * pi * r^2 ~ 0.9 * 1 => r ~ sqrt(0.9 / (50*pi)) ~ 0.075.
    # Known best is 0.0713.
    
    # Generate a hex grid that covers the square
    # Spacing s = 2*r.
    # Let's try a few different initial configurations
    
    def get_hex_grid(shift_x=0, shift_y=0, rot=0.0, scale=1.0):
        # Create a grid of points in hex pattern
        # We want about 50 points.
        # 7x7 = 49, 8x8=64. Let's aim for a 7x7 or 8x8 subset.
        # Hex spacing dx = s, dy = s*sqrt(3)/2
        
        # We will generate a large grid and then select the best 50 that fit in the square
        # Or just generate a grid that roughly fits.
        
        # Let's assume a spacing s.
        # If we use a 7x7 grid, it's 49 points. We need 50.
        # Maybe an 8x8 grid cropped?
        
        # Let's try to construct a specific hex pattern.
        # Rows of 8, 7, 8, 7, 8, 7, 8, 7 ? 8+7+8+7+8+7+8+7 = 60.
        # Rows of 7, 8, 7, 8, 7, 8, 7, 8 ? 60.
        
        # Let's just generate a hex lattice with spacing 0.15 (approx r=0.075)
        s = 0.14 # spacing between centers
        dy = s * np.sqrt(3) / 2
        
        xs = []
        ys = []
        
        # Generate points in a range larger than the square
        for row in range(-5, 10):
            y = row * dy
            for col in range(-5, 10):
                x = col * s
                if row % 2 != 0:
                    x += s / 2.0
                
                # Apply rotation
                xr = x * np.cos(rot) - y * np.sin(rot)
                yr = x * np.sin(rot) + y * np.cos(rot)
                
                xs.append(xr + shift_x)
                ys.append(yr + shift_y)
        
        grid = np.column_stack((xs, ys))
        
        # Filter points that are reasonably inside [-0.5, 0.5]
        # Actually, we want to select 50 points that maximize the minimum distance.
        # For initialization, just pick the ones closest to the center or a random subset.
        
        # To ensure we have 50 points, we can pick the 50 points from this grid
        # that are closest to the origin, or randomly.
        
        # Let's pick the 50 points with the largest minimum distance to other points?
        # That's expensive.
        
        # Simple heuristic: Pick points that are inside [-0.45, 0.45] to avoid boundary issues initially.
        mask = (np.abs(grid[:, 0]) < 0.48) & (np.abs(grid[:, 1]) < 0.48)
        candidates = grid[mask]
        
        if len(candidates) < N:
             # If not enough, use all and let optimizer push them in
             candidates = grid
        
        # Randomly select N points from candidates?
        # Or select a regular subset.
        # Let's just take the first N points from a sorted list of candidates by distance to center?
        # Or better: Random selection from the valid candidates to get diversity.
        
        idx = np.random.choice(len(candidates), N, replace=False)
        return candidates[idx]

    # Try multiple initializations
    num_restarts = 20
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Vary parameters
        shift_x = np.random.uniform(-0.1, 0.1)
        shift_y = np.random.uniform(-0.1, 0.1)
        rot = np.random.uniform(0, np.pi/6)
        
        init_centers = get_hex_grid(shift_x, shift_y, rot)
        
        # Ensure finite values
        if not np.all(np.isfinite(init_centers)):
            continue
        
        # Flatten for optimizer
        x0 = init_centers.flatten()
        
        # Bounds: centers must be in [-0.5, 0.5]
        bounds = [(-0.5, 0.5)] * (N * 2)
        
        # Run L-BFGS-B
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success:
                opt_centers = res.x.reshape(N, 2)
                # Clip to bounds just in case
                opt_centers = np.clip(opt_centers, -0.5, 0.5)
                
                r = calculate_radius(opt_centers)
                
                if r > best_radius:
                    best_radius = r
                    best_centers = opt_centers
                    
        except Exception as e:
            pass # Ignore errors in individual runs

    # If no optimization succeeded, return a basic grid
    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers

# EVOLVE-BLOCK-END
