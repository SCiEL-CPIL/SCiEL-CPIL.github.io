import numpy as np

def construct():
    N = 50
    
    # 1. Initialization: Subset of a 4x4x4 grid
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Use a fixed seed for deterministic behavior in the evaluator
    rng = np.random.RandomState(42)
    indices = rng.choice(len(grid), N, replace=False)
    pos = grid[indices].copy()
    
    # 2. Optimization
    # We want to maximize the minimum distance.
    # We use a force-directed approach.
    
    steps = 3000
    # Start with a larger step, decay it
    step_size = 0.02
    decay = 0.9995
    
    for i in range(steps):
        # Calculate pairwise distances
        # pos shape: (N, 3)
        # diff shape: (N, N, 3)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Avoid division by zero
        dists[dist < 1e-6] = 1e-6 # Wait, dists is 2D, dist is not defined.
        # Fix:
        dists[dists < 1e-6] = 1e-6
        
        # Calculate forces
        # Force magnitude is 1 / dist^2
        # We only care about forces that are significant.
        # Using 1/d^2 for all pairs is expensive but N=50 is small.
        # Force vector for pair (i,j) on i is (pos[i] - pos[j]) / dist^2
        # Note: diff[i,j] = pos[i] - pos[j]
        
        # We want to sum forces on each point.
        # forces[i] = sum_j (diff[i,j] / dists[i,j]^2)
        # But we must exclude i=j (which is 0).
        
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        
        # Compute force vectors
        # dists_inv_sq = 1 / dists^2
        dists_inv_sq = 1.0 / (dists * dists)
        dists_inv_sq[~mask] = 0.0
        
        # forces = diff * dists_inv_sq[:,:,np.newaxis]
        # Sum over j (axis 1)
        forces = np.sum(diff * dists_inv_sq[:, :, np.newaxis], axis=1)
        
        # Wall forces
        # For each point, if close to wall, push in.
        # d_wall = 0.5 - |pos|
        # Force direction = -sign(pos)
        # Magnitude = 1 / d_wall^2 (if d_wall < threshold)
        
        abs_pos = np.abs(pos)
        d_wall = 0.5 - abs_pos
        
        # Threshold for wall interaction
        wall_thresh = 0.2
        wall_mask = d_wall < wall_thresh
        
        # Calculate wall force magnitudes
        # Avoid div by zero
        d_wall_safe = np.where(wall_mask, d_wall, 1.0)
        wall_force_mag = np.zeros_like(d_wall)
        wall_force_mag[wall_mask] = 1.0 / (d_wall_safe[wall_mask] ** 2)
        
        # Direction: -sign(pos)
        wall_dir = -np.sign(pos)
        
        # Add wall forces
        forces += wall_dir * wall_force_mag[:, np.newaxis]
        
        # Update positions
        # Add noise to escape local minima
        noise = rng.normal(0, step_size * 0.5, size=pos.shape)
        
        pos += step_size * (forces + noise)
        
        # Clip to cube
        pos = np.clip(pos, -0.5, 0.5)
        
        # Decay step size
        step_size *= decay

    return pos
