"""Circle packing in unit square: pack 26 circles, maximize sum of radii.
Strategy: Physics-based bubble relaxation with explicit time-stepping.
"""
import numpy as np
import time

N = 26

def _eval_packing(pos):
    """
    Evaluate a packing configuration (N x 3 array of [x, y, r]).
    Returns (total_radius, is_valid).
    """
    xs, ys, rs = pos[:, 0], pos[:, 1], pos[:, 2]
    total_r = np.sum(rs)
    
    # Check boundaries and radii
    if np.any(rs < 0) or np.any(xs < 0) or np.any(xs > 1) or np.any(ys < 0) or np.any(ys > 1):
        return total_r, False
    if np.any(rs > xs + 1e-9) or np.any(rs > 1 - xs + 1e-9) or np.any(rs > ys + 1e-9) or np.any(rs > 1 - ys + 1e-9):
        return total_r, False
    
    # Check non-overlap
    # Vectorized pairwise check
    dx = xs[:, np.newaxis] - xs[np.newaxis, :]
    dy = ys[:, np.newaxis] - ys[np.newaxis, :]
    dists = np.sqrt(dx * dx + dy * dy)
    sum_rs = rs[:, np.newaxis] + rs[np.newaxis, :]
    
    # Exclude self (diagonal)
    np.fill_diagonal(dists, np.inf)
    
    # Check if any distance is less than sum of radii (with tolerance)
    if np.any(dists < sum_rs - 1e-8):
        return total_r, False
        
    return total_r, True

def _physics_relax(pos_init, deadline, rng):
    """
    Physics-based relaxation.
    Model:
    - Each circle is a "bubble" with internal pressure P_i proportional to r_i^2 (or simply target radius).
    - Actually, simpler model:
      Force on center i:
      1. Expansion force: pushes center away from boundaries proportional to distance to wall? No, that's repulsion from walls.
      Let's use a standard soft body physics approach:
      
      Forces:
      1. Repulsion between circles: F_rep = k * (r_i + r_j - d_ij) * direction_ij if d_ij < r_i + r_j
      2. Repulsion from walls: F_wall = k_wall * (r_i - dist_to_wall) * normal if dist_to_wall < r_i
      3. "Growth" force: To maximize sum of radii, we can simulate radii increasing over time, 
         or fix radii and optimize positions? 
         The task says "maximize sum of their radii". 
         Usually, in packing, we fix N and maximize min radius or sum radii.
         Here, radii are variables.
         
      Alternative Physics Model:
      Imagine the circles are balloons. We inflate them.
      State: (x_i, y_i, r_i).
      Forces:
      - Repulsion between circles: If overlapping, push centers apart.
      - Repulsion from walls: If overlapping wall, push center inward.
      - Inflation: r_i increases at a constant rate? 
        If we just inflate, they will expand until they touch each other and walls.
        This naturally maximizes the sum of radii (or at least the minimum radius, and likely the sum if uniform expansion is allowed).
        
      Let's try:
      dr/dt = growth_rate (constant for all, or proportional to available space?)
      If dr/dt is constant, all circles grow at same rate.
      The system will reach a state where no circle can grow further without overlap.
      This is a valid packing.
      
      To maximize SUM of radii, uniform growth might not be optimal if circles have different sizes.
      However, starting from small equal circles and growing them uniformly often leads to good packings.
      We can add a "pressure" term: 
      d(r_i)/dt = base_growth + k * (local_free_space)
      
      Simpler: 
      1. Start with small circles.
      2. Increase all radii by dt.
      3. Resolve overlaps by moving centers (Verlet integration or simple force projection).
      4. Repeat until no more growth is possible or time runs out.
      
      Let's implement a "growth and relaxation" loop.
    """
    pos = pos_init.copy()
    xs = pos[:, 0]
    ys = pos[:, 1]
    rs = pos[:, 2]
    
    # Pre-compute pair indices for efficiency
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    # Parameters
    dt_growth = 0.001
    dt_move = 0.01
    k_rep = 50.0  # Repulsion stiffness
    k_wall = 50.0 # Wall stiffness
    damping = 0.9
    max_steps = 5000
    
    velocities = np.zeros((N, 2))
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        # 1. Grow radii
        # Try to grow all circles slightly
        # Check if growth is possible
        # For simplicity, just grow and let physics resolve overlaps
        rs += dt_growth
        
        # 2. Compute forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Pairwise repulsion
        dx = xs[i_idx] - xs[j_idx]
        dy = ys[i_idx] - ys[j_idx]
        dists = np.sqrt(dx * dx + dy * dy)
        sum_rs = rs[i_idx] + rs[j_idx]
        
        # Overlap amount
        overlap = np.maximum(0, sum_rs - dists)
        
        # Only apply force if overlapping
        mask = overlap > 0
        if np.any(mask):
            # Normalize direction
            # Avoid division by zero
            safe_dists = np.where(dists > 1e-6, dists, 1e-6)
            nx = dx[mask] / safe_dists[mask]
            ny = dy[mask] / safe_dists[mask]
            
            # Force magnitude
            f_mag = k_rep * overlap[mask]
            
            # Apply force
            # Vectorized update using np.add.at
            np.add.at(fx, i_idx[mask], f_mag * nx)
            np.add.at(fy, i_idx[mask], f_mag * ny)
            np.add.at(fx, j_idx[mask], -f_mag * nx)
            np.add.at(fy, j_idx[mask], -f_mag * ny)
            
        # Wall repulsion
        # Left wall
        overlap_l = np.maximum(0, rs - xs)
        if np.any(overlap_l > 0):
            fx += k_wall * overlap_l
        # Right wall
        overlap_r = np.maximum(0, rs - (1 - xs))
        if np.any(overlap_r > 0):
            fx -= k_wall * overlap_r
        # Bottom wall
        overlap_b = np.maximum(0, rs - ys)
        if np.any(overlap_b > 0):
            fy += k_wall * overlap_b
        # Top wall
        overlap_t = np.maximum(0, rs - (1 - ys))
        if np.any(overlap_t > 0):
            fy -= k_wall * overlap_t
        
        # 3. Integrate
        # Semi-implicit Euler
        velocities = velocities * damping + (fx + fy * 0) * dt_move # Wait, fx is x-force, fy is y-force
        velocities[:, 0] = velocities[:, 0] * damping + fx * dt_move
        velocities[:, 1] = velocities[:, 1] * damping + fy * dt_move
        
        xs += velocities[:, 0]
        ys += velocities[:, 1]
        
        # Clip positions to box (hard constraint for stability, force handles soft)
        # Actually, if we clip, we might lose energy. But it prevents escape.
        # Let's just clip to [0, 1] to ensure they stay in the square.
        # The wall forces should keep them away from walls if k_wall is high enough.
        # But if they touch wall, dist=0, force is max.
        
        # Update pos
        pos[:, 0] = xs
        pos[:, 1] = ys
        pos[:, 2] = rs
        
        # Check validity
        total, valid = _eval_packing(pos)
        
        # If we have a valid packing, record it.
        # We want to keep the best valid packing found.
        # Note: During simulation, overlaps are allowed (soft), so valid will be False often.
        # We should check validity periodically or at the end.
        # Let's check every 100 steps.
        if step % 100 == 0:
            _, is_valid = _eval_packing(pos)
            if is_valid:
                # We have a valid packing. 
                # We can try to grow more.
                pass # Logic handled by loop continuing
        
        # Stopping criterion: If forces are very small and no growth is happening?
        # Since we always grow rs, we might eventually get stuck with huge overlaps.
        # We should stop if the packing becomes invalid and we can't resolve it?
        # Actually, the goal is to find a VALID packing with max sum radii.
        # If we grow too much, it becomes invalid.
        # So we should grow, relax, and if it becomes invalid, we might need to shrink?
        # No, we want to maximize.
        # Strategy: 
        # 1. Grow.
        # 2. Relax positions.
        # 3. Check if valid.
        # 4. If valid, keep growing.
        # 5. If invalid, try to relax harder? Or stop?
        # If it's invalid, it means overlaps exist. The forces will try to separate them.
        # If they can't separate, the radii are too big.
        # So we should stop growing if we can't achieve a valid configuration.
        
        # Let's track the best valid total radius found so far.
        # If we exceed the current best, update.
        # If we are invalid for a long time, maybe we grew too fast.
        
        # Simple heuristic: 
        # If the sum of forces is very small but there are still overlaps, we are stuck.
        # Or, if we are invalid, we should not grow further.
        
        _, is_valid = _eval_packing(pos)
        if not is_valid:
            # If invalid, shrink radii slightly to allow relaxation?
            # Or just stop growing?
            # Let's try to shrink back to the last valid state? 
            # That's complex. 
            # Alternative: Only grow if the previous step was valid.
            # If previous was invalid, don't grow, just relax.
            # If relaxation fails to make it valid, we are stuck.
            
            # For now, let's just not grow if invalid.
            # But we already grew in step 1.
            # Let's revert the growth if it leads to immediate invalidity that can't be fixed?
            
            # Let's try a different approach:
            # Keep a "last_valid" state.
            # If current is invalid, revert to last_valid and stop?
            # Or try to fix it.
            
            # Let's just check validity. If invalid, we won't update the "best" found.
            # We continue relaxing to see if it becomes valid.
            # If it doesn't become valid after some steps, we might be stuck.
            
            # To be safe, if we are invalid, we shrink radii by half the growth step.
            rs -= dt_growth / 2
            pos[:, 2] = rs
            
    return pos

def _local_search_polish(pos, deadline, rng):
    """
    Use a simple local search to polish the result from physics.
    For each circle, try to move it slightly to allow radius increase.
    """
    pos = pos.copy()
    best_total, _ = _eval_packing(pos)
    
    for _ in range(100):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            # Try to increase radius of circle i
            # Max possible radius given current positions of others
            # r_max_i = min(
            #   min_j (dist(i,j) - r_j),
            #   min(x_i, 1-x_i, y_i, 1-y_i)
            # )
            others_idx = [j for j in range(N) if j != i]
            if not others_idx:
                continue
            others_xy = pos[others_idx, :2]
            others_r = pos[others_idx, 2]
            
            dx = pos[i, 0] - others_xy[:, 0]
            dy = pos[i, 1] - others_xy[:, 1]
            dists = np.sqrt(dx * dx + dy * dy)
            circle_clear = np.min(dists - others_r)
            wall_clear = min(pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1])
            
            new_r = min(wall_clear, circle_clear)
            
            if new_r > pos[i, 2] + 1e-6:
                pos[i, 2] = new_r
                improved = True
        
        if not improved:
            break
            
    return pos

def construct():
    deadline = time.time() + 140  # 10s safety margin
    
    # Initial configuration: Grid
    side = 5 # 5x5 grid = 25 circles, need 26.
    # Let's use 5x6 grid? 30 circles. We take 26.
    # Or 5x5 with one in the middle?
    # Simple: 5x5 grid, remove 4 corners? No, just use 5x5 and add one?
    # Let's use a perturbed grid of 26 points.
    
    # 5x5 grid centers:
    # 0.1, 0.3, 0.5, 0.7, 0.9
    # 25 points.
    # Add one at 0.5, 0.5? No, that's occupied.
    # Let's use 5 rows, 6 cols?
    # Cols: 0.1, 0.28, 0.46, 0.64, 0.82, 1.0? No.
    
    # Let's just generate 26 random points and let physics do the work.
    # Random init is robust for physics.
    rng = np.random.default_rng(42)
    
    pos_init = np.zeros((N, 3))
    pos_init[:, 0] = rng.uniform(0.1, 0.9, N)
    pos_init[:, 1] = rng.uniform(0.1, 0.9, N)
    pos_init[:, 2] = 0.01 # Small initial radius
    
    # Run physics relaxation
    pos_phys = _physics_relax(pos_init, deadline - 10, rng)
    
    # Polish
    pos_final = _local_search_polish(pos_phys, deadline, rng)
    
    # Ensure validity
    total, valid = _eval_packing(pos_final)
    if not valid:
        # Fallback: simple grid
        # 5x5 grid
        side = 5
        pos_fallback = np.zeros((N, 3))
        count = 0
        for i in range(side):
            for j in range(side):
                if count < N:
                    pos_fallback[count, 0] = (i + 0.5) / side
                    pos_fallback[count, 1] = (j + 0.5) / side
                    pos_fallback[count, 2] = 0.5 / side - 0.01
                    count += 1
        # Clip
        for i in range(N):
            pos_fallback[i, 2] = min(pos_fallback[i, 2], pos_fallback[i, 0], 1-pos_fallback[i, 0], pos_fallback[i, 1], 1-pos_fallback[i, 1])
        return pos_fallback
        
    return pos_final

if __name__ == "__main__":
    res = construct()
    total, valid = _eval_packing(res)
    print(f"Score: {total}, Valid: {valid}")
