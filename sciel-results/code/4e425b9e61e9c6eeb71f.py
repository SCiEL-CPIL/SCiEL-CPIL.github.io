import numpy as np
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    def get_forces(xy, r, dt=0.01):
        """
        Calculate forces on centers and radii.
        Returns: force_xy (n, 2), force_r (n,)
        """
        n = N
        force_xy = np.zeros((n, 2))
        force_r = np.zeros(n)
        
        # 1. Boundary Forces (Spring-like)
        # If circle goes out of bounds, push it back
        # Strength is proportional to penetration depth
        k_boundary = 50.0
        
        # Left
        pen_l = r - xy[:, 0]
        mask = pen_l > 0
        force_xy[mask, 0] += k_boundary * pen_l[mask]
        
        # Right
        pen_r = xy[:, 0] + r - 1.0
        mask = pen_r > 0
        force_xy[mask, 0] -= k_boundary * pen_r[mask]
        
        # Bottom
        pen_b = r - xy[:, 1]
        mask = pen_b > 0
        force_xy[mask, 1] += k_boundary * pen_b[mask]
        
        # Top
        pen_t = xy[:, 1] + r - 1.0
        mask = pen_t > 0
        force_xy[mask, 1] -= k_boundary * pen_t[mask]
        
        # 2. Overlap Forces (Repulsive)
        # Pairwise interactions
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist_sq = dx**2 + dy**2
        # Avoid division by zero
        dist_sq = np.maximum(dist_sq, 1e-12)
        dist = np.sqrt(dist_sq)
        
        min_dist = r[:, None] + r[None, :]
        
        # Overlap amount
        overlap = min_dist - dist
        
        # Only consider pairs that are overlapping or very close
        # We use a soft repulsion that is very strong when overlapping
        k_repulse = 100.0
        
        # Force magnitude is proportional to overlap
        # Direction is along the line connecting centers
        # F = k * overlap * unit_vector
        
        # Unit vector: (dx, dy) / dist
        # Note: dx[i,j] = x_i - x_j. So force on i is in direction (x_i - x_j) if we want to push i away from j.
        # Wait, if overlap > 0, we want to push i away from j.
        # Vector from j to i is (x_i - x_j).
        # So force on i should be in direction (x_i - x_j) / dist.
        
        # Create force matrix
        # F_ij is force on i due to j
        # F_ij = k * overlap_ij * (p_i - p_j) / dist_ij
        
        # We only care about i != j
        # dist matrix has 0 on diagonal, we set it to 1e-12 so 1/dist is large, but overlap on diagonal is r_i+r_i - 0 = 2r_i > 0.
        # We must mask out diagonal.
        
        mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask, False)
        
        # Apply only where overlap > 0
        active = (overlap > 0) & mask
        
        # Calculate force vector components
        # f_x = k * overlap * dx / dist
        # f_y = k * overlap * dy / dist
        
        f_mag = k_repulse * overlap
        # Set to 0 where not active
        f_mag[~active] = 0.0
        
        f_x = f_mag * (dx / dist)
        f_y = f_mag * (dy / dist)
        
        # Sum forces from all other circles
        force_xy[:, 0] += np.sum(f_x, axis=1)
        force_xy[:, 1] += np.sum(f_y, axis=1)
        
        # 3. Radius Growth Force
        # We want to maximize sum of radii.
        # This is equivalent to applying a constant outward force on radii.
        # However, we must ensure the system remains stable.
        # If we just grow radii, they will overlap and push each other.
        # The equilibrium will be where repulsion balances growth.
        # But simple constant growth might lead to explosion if not careful.
        # Better: Grow radii only if there is "space" or just apply a small constant force 
        # and let the repulsion handle the rest.
        # Actually, a better strategy for packing is to keep radii fixed and move centers?
        # No, the task is to maximize sum of radii.
        # If we fix radii, we can't improve.
        # So we must grow them.
        
        # Let's use a small constant growth force.
        # The repulsion will naturally limit the growth when the square is full.
        force_r += 0.01  # Constant growth drive
        
        # Damping of radius growth if close to max possible? 
        # Max radius is 0.5. 
        # Let's add a soft penalty if r > 0.45 to prevent numerical issues, though unlikely.
        # Actually, just the repulsion should suffice.
        
        return force_xy, force_r

    def simulate(xy_init, r_init, n_steps=5000, dt=0.005, damping=0.9):
        """
        Run physics simulation.
        """
        xy = xy_init.copy()
        r = r_init.copy()
        
        # Velocities
        v_xy = np.zeros_like(xy)
        v_r = np.zeros_like(r)
        
        best_score = -np.inf
        best_xy = xy.copy()
        best_r = r.copy()
        
        for step in range(n_steps):
            if time.time() > deadline - 2:
                break
                
            # Calculate forces
            f_xy, f_r = get_forces(xy, r, dt)
            
            # Update velocities (Euler)
            v_xy += f_xy * dt
            v_r += f_r * dt
            
            # Damping (Friction)
            v_xy *= damping
            v_r *= damping
            
            # Update positions
            xy += v_xy * dt
            r += v_r * dt
            
            # Hard constraints to prevent explosion
            # Radii must be positive
            r = np.maximum(r, 0.001)
            # Radii must be <= 0.5
            r = np.minimum(r, 0.5)
            
            # Centers must be roughly in [0,1] to avoid flying away
            # If a circle flies out, the boundary force should pull it back, 
            # but let's clamp positions to [-1, 2] to be safe.
            xy = np.clip(xy, -1.0, 2.0)
            
            # Check validity and score
            # We can do a quick check every few steps
            if step % 10 == 0:
                # Quick validity check
                valid = True
                # Boundary
                if np.any(r > 0.5) or np.any(xy[:, 0] < r) or np.any(xy[:, 0] > 1 - r) or \
                   np.any(xy[:, 1] < r) or np.any(xy[:, 1] > 1 - r):
                    valid = False
                
                if valid:
                    # Check overlaps
                    dx = xy[:, 0][:, None] - xy[:, 0][None, :]
                    dy = xy[:, 1][:, None] - xy[:, 1][None, :]
                    dist = np.sqrt(dx**2 + dy**2)
                    min_dist = r[:, None] + r[None, :]
                    # Diagonal is 0, min_dist is 2r > 0, so dist < min_dist is true on diagonal.
                    # We need to exclude diagonal.
                    iu = np.triu_indices(N, k=1)
                    if np.any(dist[iu] < min_dist[iu] - 1e-9):
                        valid = False
                
                if valid:
                    score = np.sum(r)
                    if score > best_score:
                        best_score = score
                        best_xy = xy.copy()
                        best_r = r.copy()
        
        return best_xy, best_r

    def is_valid(xy, r, tol=1e-9):
        if np.any(r < -tol) or np.any(r > 0.5 + tol):
            return False
        if np.any(xy[:, 0] - r < -tol) or np.any(1 - xy[:, 0] - r < -tol):
            return False
        if np.any(xy[:, 1] - r < -tol) or np.any(1 - xy[:, 1] - r < -tol):
            return False
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(N, k=1)
        if np.any(dist[iu] < min_dist[iu] - tol):
            return False
        return True

    rng = np.random.RandomState(42)
    
    best_result = None
    best_score = -np.inf
    
    # Multiple restarts with different initial conditions
    # 1. Grid
    # 2. Random
    # 3. Hexagonal
    
    restarts = [
        "grid", "random", "hex", "grid_perturbed", "random_large"
    ]
    
    for i, restart_type in enumerate(restarts):
        if time.time() > deadline - 10:
            break
            
        if restart_type == "grid":
            side = 6
            xy = np.zeros((N, 2))
            r = np.zeros(N)
            for i in range(N):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
                r[i] = 0.07 # Start small
        elif restart_type == "random":
            xy = rng.uniform(0.1, 0.9, size=(N, 2))
            r = rng.uniform(0.05, 0.08, size=N)
        elif restart_type == "hex":
            xy = np.zeros((N, 2))
            r = np.zeros(N)
            # Hexagonal lattice
            dx = 0.18
            dy = dx * np.sqrt(3) / 2
            idx = 0
            for row in range(6):
                for col in range(6):
                    if idx >= N: break
                    x = 0.1 + col * dx + (0.09 if row % 2 == 1 else 0)
                    y = 0.1 + row * dy
                    if x < 0.95 and y < 0.95:
                        xy[idx] = [x, y]
                        r[idx] = 0.08
                        idx += 1
            # Fill remaining
            for k in range(idx, N):
                xy[k] = rng.uniform(0.1, 0.9, size=2)
                r[k] = 0.05
        elif restart_type == "grid_perturbed":
            side = 6
            xy = np.zeros((N, 2))
            r = np.zeros(N)
            for i in range(N):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5)/side + rng.uniform(-0.02, 0.02), 
                         (row + 0.5)/side + rng.uniform(-0.02, 0.02)]
                r[i] = 0.07
        else: # random_large
            xy = rng.uniform(0.15, 0.85, size=(N, 2))
            r = rng.uniform(0.08, 0.12, size=N)
        
        # Run simulation
        # We run a few simulations with different dt or damping? 
        # Let's just run one long simulation per restart.
        # To escape local minima, we can add a small random noise to forces occasionally?
        # Or just rely on the multiple restarts.
        
        sim_xy, sim_r = simulate(xy, r, n_steps=2000, dt=0.005, damping=0.95)
        
        if is_valid(sim_xy, sim_r):
            score = np.sum(sim_r)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([sim_xy, sim_r])
        
        # If we have time, do a second pass with slightly different parameters
        if time.time() < deadline - 15 and best_result is not None:
             # Use the best found so far as a new initial condition? 
             # Or just try another random one.
             pass

    # Final Polish: Use the best result and run a short simulation with very small dt 
    # to settle any remaining jitter, ensuring validity.
    if best_result is not None:
        xy_best = best_result[:, :2]
        r_best = best_result[:, 2]
        
        # Run a short, stable simulation to polish
        # Reduce damping to let it settle, or increase it to stop movement?
        # We want to stop movement, so high damping.
        polished_xy, polished_r = simulate(xy_best, r_best, n_steps=500, dt=0.001, damping=0.99)
        
        if is_valid(polished_xy, polished_r):
            score = np.sum(polished_r)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([polished_xy, polished_r])
        else:
            # If polish broke it, try to repair by scaling down radii slightly
            for scale in [0.99, 0.98, 0.95, 0.9]:
                r_scaled = r_best * scale
                if is_valid(xy_best, r_scaled):
                    score = np.sum(r_scaled)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack([xy_best, r_scaled])
                    break
    
    if best_result is None:
        # Fallback
        side = 6
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.07) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
