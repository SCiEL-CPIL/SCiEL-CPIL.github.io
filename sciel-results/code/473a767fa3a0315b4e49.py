import numpy as np
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit
    
    # Precompute pair indices
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    def get_forces(x, y, r, temperature):
        """
        Calculate forces on positions and radii.
        Returns:
        fx, fy: Forces on x, y coordinates
        fr: Forces on radii
        """
        fx = np.zeros(N)
        fy = np.zeros(N)
        fr = np.zeros(N)
        
        # 1. Pairwise Repulsion (Collision)
        # Force is proportional to overlap depth, scaled by stiffness
        # We use a stiff spring for overlaps to push them apart
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(dist_sq + 1e-12)
        
        # Overlap depth: r_i + r_j - dist
        overlap = (r[i_idx] + r[j_idx]) - dist
        
        # Only apply force if overlapping
        mask = overlap > 0
        if np.any(mask):
            # Stiffness increases as temperature drops (harder constraint)
            # But for stability, we keep it constant high and rely on damping
            k_repulse = 50.0
            
            # Force vector direction
            # F = k * overlap * (dx/dist, dy/dist)
            # Note: If dist is 0, direction is undefined. We handle this with small noise or skip.
            # Since we start with non-zero dist mostly, this is rare.
            
            # Apply to i
            fx_i = k_repulse * overlap[mask] * dx[mask] / dist[mask]
            fy_i = k_repulse * overlap[mask] * dy[mask] / dist[mask]
            
            # Apply to j (opposite)
            # We need to scatter these back to arrays
            # Using np.add.at for accumulated forces
            np.add.at(fx, i_idx[mask], fx_i)
            np.add.at(fy, i_idx[mask], fy_i)
            np.add.at(fx, j_idx[mask], -fx_i)
            np.add.at(fy, j_idx[mask], -fy_i)
            
            # Force on radii:
            # Overlap decreases if we shrink radii.
            # Potential energy ~ overlap^2. dE/dr ~ 2*overlap.
            # So force on r is -2*overlap (pushes r down to reduce overlap)
            # But we also have an "expansion" force.
            # Let's model the collision as a pressure that pushes radii down.
            fr_i = -2.0 * k_repulse * overlap[mask]
            fr_j = -2.0 * k_repulse * overlap[mask]
            
            np.add.at(fr, i_idx[mask], fr_i)
            np.add.at(fr, j_idx[mask], fr_j)

        # 2. Boundary Constraints (Pressure from walls)
        # We want circles to stay inside [0,1]^2.
        # If x < r, we need to push x right and r left (or x right).
        # Actually, the constraint is x >= r and 1-x >= r.
        # Let's apply a strong repulsive force from the boundaries.
        
        # Left wall: x - r < 0
        viol_left = r - x
        mask_l = viol_left > 0
        if np.any(mask_l):
            # Push x to the right
            fx[mask_l] += 50.0 * viol_left[mask_l]
            # Push r down (shrinking helps satisfy x>=r)
            fr[mask_l] -= 50.0 * viol_left[mask_l]
            
        # Right wall: 1 - x - r < 0  => x + r > 1
        viol_right = x + r - 1.0
        mask_r = viol_right > 0
        if np.any(mask_r):
            # Push x to the left
            fx[mask_r] -= 50.0 * viol_right[mask_r]
            # Push r down
            fr[mask_r] -= 50.0 * viol_right[mask_r]
            
        # Bottom wall: y - r < 0
        viol_bottom = r - y
        mask_b = viol_bottom > 0
        if np.any(mask_b):
            fy[mask_b] += 50.0 * viol_bottom[mask_b]
            fr[mask_b] -= 50.0 * viol_bottom[mask_b]
            
        # Top wall: 1 - y - r < 0
        viol_top = y + r - 1.0
        mask_t = viol_top > 0
        if np.any(mask_t):
            fy[mask_t] -= 50.0 * viol_top[mask_t]
            fr[mask_t] -= 50.0 * viol_top[mask_t]

        # 3. Expansion Force (Objective: Maximize Sum(r))
        # We apply a constant "pressure" to grow radii.
        # This balances against the repulsion from walls and other circles.
        # The magnitude of this force determines the final density.
        # If too high, circles will try to grow and push into walls/each other,
        # resulting in high penalty/overlap. The equilibrium is where expansion
        # force balances repulsion.
        # We tune this to be strong enough to maximize size but not so strong
        # that it causes instability.
        expansion_force = 0.5
        fr += expansion_force
        
        return fx, fy, fr

    def project_valid(x, y, r):
        """
        Hard projection to ensure validity for the final output.
        Iteratively shrinks radii to resolve overlaps and boundary violations.
        """
        # Boundary clipping
        max_r_bound = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
        r = np.minimum(r, max_r_bound)
        r = np.maximum(r, 1e-6)
        
        # Pairwise resolution
        for _ in range(50):
            dx = x[i_idx] - x[j_idx]
            dy = y[i_idx] - y[j_idx]
            dist = np.sqrt(dx*dx + dy*dy + 1e-12)
            sum_r = r[i_idx] + r[j_idx]
            violations = sum_r - dist
            
            if np.max(violations) < 1e-9:
                break
            
            # Shrink radii involved in violations
            # Simple approach: shrink all radii slightly if any violation exists
            # More targeted: shrink only those in violation
            # For simplicity and robustness, global shrink is safer but slower to converge.
            # Let's try targeted:
            # If pair (i,j) violates, reduce r_i and r_j.
            # We can just do a global scale factor based on max violation.
            scale = 1.0 - np.max(violations) * 0.5
            if scale < 0.9: scale = 0.9
            r *= scale
            r = np.minimum(r, max_r_bound)
            
        return r

    # --- Main Simulation Loop ---
    
    # Initial Configuration: Hexagonal-like grid
    # This is a good starting point for dense packing
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.full(N, 0.05)
    
    # 5x5 grid with hex offset
    rows = 5
    cols = 5
    spacing = 0.18
    offset_x = 0.1
    offset_y = 0.1
    count = 0
    for i in range(rows):
        for j in range(cols):
            if count < N:
                x[count] = offset_x + j * spacing + (i % 2) * (spacing / 2)
                y[count] = offset_y + i * spacing
                count += 1
    
    # Ensure initial validity
    r = project_valid(x, y, r)
    
    # Velocity arrays
    vx = np.zeros(N)
    vy = np.zeros(N)
    vr = np.zeros(N)
    
    # Simulation parameters
    dt = 0.01
    damping = 0.9
    # Temperature schedule: Start high to allow rearrangement, cool down
    T_start = 1.0
    T_end = 0.01
    total_steps = 5000
    steps_per_check = 100
    
    best_score = np.sum(r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    current_step = 0
    
    while current_step < total_steps:
        # Check time
        if time.time() > deadline:
            break
            
        # Annealing schedule: Linear cooling
        progress = current_step / total_steps
        temperature = T_start + (T_end - T_start) * progress
        
        # Get forces
        fx, fy, fr = get_forces(x, y, r, temperature)
        
        # Update velocities (Langevin dynamics)
        # v_new = damping * v_old + dt * F + random_kick
        # Random kick strength depends on temperature
        kick_strength = 0.05 * temperature
        
        vx = damping * vx + dt * fx + np.random.normal(0, kick_strength, N)
        vy = damping * vy + dt * fy + np.random.normal(0, kick_strength, N)
        vr = damping * vr + dt * fr + np.random.normal(0, kick_strength, N)
        
        # Update positions
        x += dt * vx
        y += dt * vy
        r += dt * vr
        
        # Hard constraints during simulation (to prevent explosion)
        # Clip positions to [0, 1]
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        
        # Clip radii to [0, 0.5]
        r = np.clip(r, 0.001, 0.5)
        
        # Optional: Project to validity every few steps to keep system stable?
        # No, that kills the dynamics. Let the forces handle it.
        # But if r becomes too large and causes massive overlaps, the forces
        # will push it back.
        
        # Evaluate score
        # We only care about valid configurations for the score.
        # However, during simulation, we might have overlaps.
        # We track the best *valid* configuration.
        # To check validity efficiently, we can check max overlap.
        
        if current_step % steps_per_check == 0:
            # Check validity
            dx = x[i_idx] - x[j_idx]
            dy = y[i_idx] - y[j_idx]
            dist = np.sqrt(dx*dx + dy*dy + 1e-12)
            sum_r = r[i_idx] + r[j_idx]
            max_viol = np.max(sum_r - dist)
            
            # Boundary check
            max_viol_bound = np.max([
                np.max(r - x), np.max(x + r - 1),
                np.max(r - y), np.max(y + r - 1)
            ])
            
            if max_viol < 1e-6 and max_viol_bound < 1e-6:
                score = np.sum(r)
                if score > best_score:
                    best_score = score
                    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
            else:
                # If invalid, we can project to get a valid candidate for comparison
                # This is expensive, so we do it less frequently or only if close to valid
                # For now, let's just project if we are "close" to valid
                if max_viol < 0.05 and max_viol_bound < 0.05:
                    r_proj = project_valid(x, y, r)
                    score = np.sum(r_proj)
                    if score > best_score:
                        best_score = score
                        best_x, best_y, best_r = x.copy(), y.copy(), r_proj.copy()
                        
        current_step += 1

    # Final Polish: Use the best state found and run a few more optimization steps
    # or just project the final state if it's better
    final_r = project_valid(best_x, best_y, best_r)
    final_score = np.sum(final_r)
    
    # Check if the current (last) state is better after projection
    last_r = project_valid(x, y, r)
    last_score = np.sum(last_r)
    
    if last_score > final_score:
        best_x, best_y, best_r = x, y, last_r
        final_score = last_score
    else:
        best_r = final_r

    # Ensure final output is strictly valid
    final_r = project_valid(best_x, best_y, best_r)
    
    return np.column_stack([best_x, best_y, final_r])

# EVOLVE-BLOCK-END
