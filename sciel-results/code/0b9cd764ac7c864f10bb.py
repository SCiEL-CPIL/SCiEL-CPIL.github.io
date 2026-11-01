"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_radius(c):
    """Compute the largest common radius for centres c (shape (N, 3))."""
    if c.shape != (N, 3):
        return -1.0
    if not np.all(np.isfinite(c)):
        return -1.0
    
    # Wall distances
    abs_c = np.abs(c)
    wall_dists = 0.5 - abs_c
    min_wall = np.min(wall_dists, axis=1)
    min_wall_global = np.min(min_wall)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    return min(min_wall_global, min_dist / 2.0)

def _simulate_packing(init_pos, time_limit):
    """
    Rigid-body dynamics simulation to find a good packing.
    Uses velocity Verlet integration with adaptive damping.
    """
    # Parameters
    # Target radius is roughly 0.11. We want to push spheres apart until they settle.
    # We simulate a system where spheres repel if distance < d_target.
    # Let's set d_target slightly larger than 2*r_est to ensure they push apart.
    # r_est ~ 0.11 -> d_target ~ 0.24
    
    d_target = 0.24 
    k_rep = 500.0    # Repulsion strength
    k_wall = 5000.0  # Wall repulsion strength (stronger to keep inside)
    damping = 0.98   # Velocity damping per step
    dt = 0.005       # Time step
    
    pos = init_pos.copy()
    vel = np.zeros_like(pos)
    
    # Ensure initial positions are strictly inside
    pos = np.clip(pos, -0.49, 0.49)
    
    start_time = time.time()
    
    # Precompute indices for pairwise interactions? 
    # N=100, so 4950 pairs. Vectorizing forces is feasible but memory heavy if done naively each step.
    # We will compute forces vectorized.
    
    max_steps = 20000
    
    for step in range(max_steps):
        if time.time() > time_limit:
            break
        
        # 1. Compute Forces
        forces = np.zeros_like(pos)
        
        # Pairwise Repulsion
        # diff[i, j] = pos[i] - pos[j]
        # dist[i, j] = ||diff[i, j]||
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        
        # Avoid division by zero
        dist[dist < 1e-9] = 1e-9
        
        # Force magnitude: k_rep * (d_target - dist)^p if dist < d_target
        # Using a simple repulsive potential: F = k_rep * (d_target - dist) * (diff / dist)
        # This is a linear spring force when overlapping.
        # To make it stronger at close range, we can use (d_target - dist)^2 or similar.
        # Let's use a steeper potential: F = k_rep * (d_target - dist)^2 * dir
        overlap = np.maximum(d_target - dist, 0.0)
        # Force vector for i from j: k_rep * overlap^2 * (pos[i]-pos[j])/dist
        # We sum over j for each i.
        
        # Create matrix of force contributions
        # coeff[i, j] = k_rep * overlap[i, j]^2 / dist[i, j]
        # force[i] = sum_j coeff[i, j] * diff[i, j]
        
        coeff = np.zeros_like(dist)
        mask = overlap > 0
        coeff[mask] = k_rep * (overlap[mask] ** 2) / dist[mask]
        
        # Multiply diff by coeff and sum over j (axis 1)
        # diff shape (N, N, 3), coeff shape (N, N)
        # coeff[:, :, np.newaxis] * diff -> (N, N, 3)
        pair_forces = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
        forces += pair_forces
        
        # Wall Repulsion
        # If pos > 0.5 - margin, push left. If pos < -0.5 + margin, push right.
        # We use a soft wall that activates near the boundary.
        # Let's define a "virtual" wall at +/- 0.5.
        # Force = k_wall * (0.5 - |pos|)^p if |pos| > 0.5 - eps
        # Actually, simpler: 
        # F_x = -k_wall * (pos_x - 0.5) if pos_x > 0.5
        # But we want to keep them inside, so we apply force if they get close to the wall.
        # Let's apply force if pos > 0.45 or pos < -0.45.
        # This keeps them away from the exact boundary, allowing the radius to be determined by the min clearance.
        
        # Better: Apply force proportional to how much they "want" to cross a virtual boundary at 0.5 - r_est?
        # No, we don't know r_est. 
        # Standard approach: Harmonic potential outside the box.
        # F = -k_wall * (pos - 0.5) for pos > 0.5
        # But we want them to pack *inside*. So we want them to stay within [-0.5, 0.5].
        # If they touch the wall, their clearance to the wall is 0.
        # We want to maximize the min clearance.
        # In the simulation, we can just let them hit the walls and bounce/stop.
        # But for a smooth gradient, we use a repulsive force from the walls.
        
        # Let's use a repulsive force that increases as they approach the wall.
        # F_wall_x = k_wall * (0.5 - pos_x) for pos_x > 0.4
        # This pushes them towards the center if they are in the outer 10% of the cube.
        # This might bias them to the center, which is good for packing?
        # Actually, we want them to fill the cube. 
        # If we only repel from the *outside*, they will pack against the walls.
        # F = -k_wall * (pos - 0.5) if pos > 0.5 is for outside.
        # We want them inside. So if pos is close to 0.5, we don't push them away from the wall, 
        # we just prevent them from going outside.
        # In the dynamics, if we allow them to go slightly outside, the radius calculation will fail (or be low).
        # So we must keep them inside.
        # Let's use a hard constraint via projection after velocity update, 
        # OR a strong repulsive force from the walls.
        
        # Let's use strong repulsive force from walls at +/- 0.5.
        # F_x = -k_wall * (pos_x - 0.5) if pos_x > 0.5
        # F_x = -k_wall * (pos_x + 0.5) if pos_x < -0.5
        # This keeps them inside. But it doesn't encourage them to *use* the space near the walls.
        # However, the pairwise repulsion will push them outwards.
        # So: Pairwise repulsion pushes them out, Wall repulsion pushes them in.
        # Equilibrium will be where these balance, or where they hit the wall.
        # If they hit the wall, the wall force will be huge, keeping them just inside.
        
        # Let's implement wall forces for positions outside [-0.5, 0.5].
        # To avoid stiffness, we can start the wall force a bit earlier, e.g., at 0.45.
        # But if we do that, they won't pack against the walls, reducing the max radius.
        # Best strategy: Let them go slightly outside in the simulation? No, score becomes 0.
        # So we must keep them inside.
        # Let's use a very stiff spring for the walls.
        
        # If pos > 0.5: F = -k_wall * (pos - 0.5)
        # If pos < -0.5: F = -k_wall * (pos + 0.5)
        
        # This is only active if they cross the boundary.
        # To prevent them from crossing, we need to be careful with dt.
        # Alternatively, we can project them back into the box after each step.
        # Let's use projection for stability and simplicity, and rely on pairwise repulsion for packing.
        
        # Actually, a better physics model:
        # Treat the walls as fixed spheres of infinite radius? No.
        # Let's just project positions to the box after updating.
        # And add a small damping.
        
        # Let's refine the force calculation.
        # We already have pair_forces.
        # We don't need wall forces if we project.
        
        # 2. Update Velocity
        vel += forces * dt
        vel *= damping
        
        # 3. Update Position
        pos += vel * dt
        
        # 4. Project to Box
        pos = np.clip(pos, -0.5, 0.5)
        
        # Optional: Zero out velocity of points that hit the wall to reduce jitter
        # hit_mask = (pos > 0.499) | (pos < -0.499)
        # vel[hit_mask] = 0.0 # This is too aggressive, let's just damp.
        
    return pos

def _build_initial_grid():
    """Simple cubic grid initialization."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # We will run multiple simulations with different initializations and parameters
    best_c = None
    best_score = -1.0
    
    # Initializations
    inits = [
        _build_initial_grid(),
        np.random.uniform(-0.45, 0.45, size=(N, 3)),
        np.random.uniform(-0.45, 0.45, size=(N, 3)),
        np.random.uniform(-0.45, 0.45, size=(N, 3)),
    ]
    
    # Simulation parameters to try
    # We can vary d_target and k_rep
    configs = [
        (0.24, 500.0),
        (0.22, 800.0),
        (0.26, 400.0),
    ]
    
    # Time allocation
    # 12 inits total (4 inits * 3 configs)
    # 145s / 12 ~ 12s per sim
    
    for i, c0 in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
            
        for j, (d_t, k_r) in enumerate(configs):
            if time.time() > deadline - 5.0:
                break
            
            time_left = deadline - time.time() - 2.0
            if time_left < 1.0:
                break
            
            # Run simulation
            c_sim = _simulate_packing(c0, time.time() + time_left)
            
            # Calculate score
            score = _compute_radius(c_sim)
            
            if score > best_score:
                best_score = score
                best_c = c_sim
            
            # Local polish: A few steps of gradient ascent on the min clearance?
            # Or just return the simulation result.
            # Given the time, let's do a quick local search on the best result at the end.
            
    if best_c is None:
        return _build_initial_grid()
        
    # Final Polish: Use a simple iterative improvement on the best candidate
    # We can try to move points with low clearance away from neighbors/walls
    if time.time() < deadline - 2.0:
        c_final = best_c.copy()
        current_r = _compute_radius(c_final)
        
        # Simple local search: 
        # For a few iterations, find the point with min clearance and nudge it.
        for _ in range(500):
            if time.time() > deadline - 1.0:
                break
                
            abs_c = np.abs(c_final)
            wall_dists = 0.5 - abs_c
            min_wall_per_point = np.min(wall_dists, axis=1)
            
            # Pairwise min distance
            diff = c_final[:, np.newaxis, :] - c_final[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dist, np.inf)
            min_pair_per_point = np.min(dist, axis=1) / 2.0
            
            clearance = np.minimum(min_wall_per_point, min_pair_per_point)
            min_idx = np.argmin(clearance)
            
            # Direction to move
            if min_wall_per_point[min_idx] < min_pair_per_point[min_idx]:
                # Wall is limiting
                # Move away from closest wall
                x, y, z = c_final[min_idx]
                # Find closest wall
                d_x_min = 0.5 - x
                d_x_max = 0.5 + x
                d_y_min = 0.5 - y
                d_y_max = 0.5 + y
                d_z_min = 0.5 - z
                d_z_max = 0.5 + z
                dists_to_walls = np.array([d_x_min, d_x_max, d_y_min, d_y_max, d_z_min, d_z_max])
                min_d_idx = np.argmin(dists_to_walls)
                dir_vec = np.zeros(3)
                if min_d_idx == 0: dir_vec[0] = 1.0
                elif min_d_idx == 1: dir_vec[0] = -1.0
                elif min_d_idx == 2: dir_vec[1] = 1.0
                elif min_d_idx == 3: dir_vec[1] = -1.0
                elif min_d_idx == 4: dir_vec[2] = 1.0
                elif min_d_idx == 5: dir_vec[2] = -1.0
            else:
                # Pair is limiting
                # Move away from nearest neighbor
                nn_idx = np.argmin(np.delete(dist[min_idx], min_idx))
                # Reconstruct full index
                if nn_idx >= min_idx:
                    nn_idx += 1
                dir_vec = c_final[min_idx] - c_final[nn_idx]
                norm = np.linalg.norm(dir_vec)
                if norm < 1e-9:
                    dir_vec = np.random.normal(size=3)
                    norm = np.linalg.norm(dir_vec)
                dir_vec /= norm
            
            # Small step
            step = 0.001
            new_pos = c_final[min_idx] + dir_vec * step
            # Clip to box
            new_pos = np.clip(new_pos, -0.5, 0.5)
            
            c_new = c_final.copy()
            c_new[min_idx] = new_pos
            
            new_r = _compute_radius(c_new)
            if new_r > current_r + 1e-10:
                c_final = c_new
                current_r = new_r
            else:
                # Try smaller step or random jitter?
                # Just break if stuck? Or try another point?
                # For now, break to save time.
                break
        
        if current_r > best_score:
            best_c = c_final
            best_score = current_r

    return best_c
# EVOLVE-BLOCK-END
