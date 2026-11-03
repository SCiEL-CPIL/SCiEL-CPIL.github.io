# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct centers of 300 equal circles in the unit disk using an improved physics-based simulation.
    Improvements:
    1. Jittered hexagonal initialization to break symmetry.
    2. Steeper repulsive potential for tighter packing.
    3. Adaptive time-stepping and damping.
    4. Hard boundary constraints via projection.
    """
    N = 300
    
    # 1. Initialization: Jittered Hex Grid
    # Estimate spacing
    s = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
    
    points = []
    max_idx = int(1.5 / s) + 2
    
    for j in range(-max_idx, max_idx + 1):
        for i in range(-max_idx, max_idx + 1):
            x = i * s + (j % 2) * s / 2
            y = j * s * np.sqrt(3) / 2
            if x*x + y*y <= 0.9: # Keep slightly inside for jitter
                points.append([x, y])
                
    points = np.array(points)
    
    # Select N points
    if len(points) > N:
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)
        points = points[indices[:N]]
    elif len(points) < N:
        # Fallback
        points = np.random.uniform(-1, 1, (N, 2))
        mask = points[:,0]**2 + points[:,1]**2 <= 1
        while mask.sum() < N:
            new_pts = np.random.uniform(-1, 1, (N, 2))
            mask = new_pts[:,0]**2 + new_pts[:,1]**2 <= 1
            points = np.vstack([points, new_pts[mask]])
        points = points[:N]
        
    # Add small jitter to break symmetry
    jitter = 0.01 * s
    points = points + np.random.uniform(-jitter, jitter, points.shape)
    
    positions = points
    velocities = np.zeros_like(positions)
    
    # 2. Simulation Parameters
    # Target radius for force calculation. We want to maximize min_dist.
    # We simulate with a "virtual" radius r_sim. If the system settles,
    # the actual min_dist will be close to 2*r_sim if forces are balanced.
    # Using a slightly larger r_sim in the force law encourages spreading.
    r_sim = 0.0535 # Slightly higher than known optimum to push for expansion
    
    # Force constants
    k_rep = 1000.0
    power = 3 # Steep potential: F ~ (2r - d)^power
    dt = 0.01
    damping = 0.9
    max_iter = 5000
    
    # Spatial Hashing
    cell_size = 2 * r_sim * 1.5
    grid_offset = 1.5
    
    for step in range(max_iter):
        # Adaptive dt and damping
        # Cool down as we progress
        if step < 1000:
            dt_cur = 0.01
            damp_cur = 0.9
        elif step < 3000:
            dt_cur = 0.005
            damp_cur = 0.95
        else:
            dt_cur = 0.002
            damp_cur = 0.98
            
        # Build spatial hash grid
        grid = {}
        for i in range(N):
            x, y = positions[i]
            cx = int((x + grid_offset) / cell_size)
            cy = int((y + grid_offset) / cell_size)
            key = (cx, cy)
            if key not in grid:
                grid[key] = []
            grid[key].append(i)
            
        forces = np.zeros_like(positions)
        
        # Calculate inter-particle forces
        for i in range(N):
            x, y = positions[i]
            cx = int((x + grid_offset) / cell_size)
            cy = int((y + grid_offset) / cell_size)
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    key = (cx + dx, cy + dy)
                    if key in grid:
                        for j in grid[key]:
                            if i == j:
                                continue
                            
                            dx_pos = positions[i][0] - positions[j][0]
                            dy_pos = positions[i][1] - positions[j][1]
                            d_sq = dx_pos*dx_pos + dy_pos*dy_pos
                            
                            # Interaction range: slightly larger than 2*r_sim to be safe
                            if d_sq < (2 * r_sim * 1.2)**2 and d_sq > 1e-12:
                                d = np.sqrt(d_sq)
                                # Repulsive force
                                # Potential V = k * (2r - d)^power
                                # Force F = -dV/dd = k * power * (2r - d)^(power-1)
                                # Direction: away from j
                                if d < 2 * r_sim:
                                    f_mag = k_rep * (2 * r_sim - d) ** power
                                    f_x = f_mag * dx_pos / d
                                    f_y = f_mag * dy_pos / d
                                    forces[i][0] += f_x
                                    forces[i][1] += f_y
                                else:
                                    # Weak attraction or no force? 
                                    # For packing, we just want to avoid overlap.
                                    # No attraction is better to allow free movement.
                                    pass
                                
        # Boundary forces: Use a soft boundary first, then hard project
        dist_center = np.linalg.norm(positions, axis=1)
        limit = 1.0 - r_sim
        overflow = np.maximum(0, dist_center - limit)
        safe_dist = np.where(dist_center > 1e-9, dist_center, 1.0)
        
        # Soft boundary force
        boundary_force_mag = 500.0 * overflow**2
        boundary_force_vec = boundary_force_mag / safe_dist
        forces[:, 0] -= boundary_force_vec * positions[:, 0]
        forces[:, 1] -= boundary_force_vec * positions[:, 1]
        
        # Update velocities and positions
        velocities = velocities * damp_cur + forces * dt_cur
        positions = positions + velocities * dt_cur
        
        # Hard Boundary Constraint: Project back if outside
        dist_center = np.linalg.norm(positions, axis=1)
        max_dist = 1.0 - 0.01 # Keep slightly inside to avoid numerical issues at boundary
        outside_mask = dist_center > max_dist
        if np.any(outside_mask):
            # Scale positions back to max_dist
            scale = max_dist / dist_center[outside_mask]
            positions[outside_mask] *= scale[:, np.newaxis]
            # Zero out radial velocity component for outside particles
            # v_radial = (v . r_hat)
            r_hat = positions[outside_mask] / dist_center[outside_mask, np.newaxis]
            v_radial = np.sum(velocities[outside_mask] * r_hat, axis=1)
            velocities[outside_mask] -= v_radial[:, np.newaxis] * r_hat

        # Early stopping
        if step > 1000 and np.max(np.linalg.norm(velocities, axis=1)) < 1e-5:
            break
            
    return positions
# EVOLVE-BLOCK-END
