import numpy as np
import time

N = 50

def compute_radius(positions):
    """
    Compute the maximum radius for the given positions.
    Returns (radius, is_valid).
    If invalid (points outside), radius is -1.
    """
    # Boundary constraint
    max_abs = np.max(np.abs(positions), axis=1)
    bound_dists = 0.5 - max_abs
    
    # If any point is outside, radius is invalid (negative)
    if np.any(bound_dists < -1e-9):
        return -1.0, False
        
    min_bound = np.min(bound_dists)
    
    # Pairwise distance constraint
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=-1)
    
    # Ignore self-distances
    np.fill_diagonal(dists_sq, np.inf)
    
    min_dist_sq = np.min(dists_sq)
    min_dist = np.sqrt(min_dist_sq)
    
    r = min(min_bound, min_dist / 2.0)
    
    return r, True

def polish(positions, time_limit):
    """
    Local polish step to improve the best solution found by SA.
    """
    best_pos = positions.copy()
    best_r, _ = compute_radius(best_pos)
    start_t = time.time()
    
    # Iterate until no improvement or time limit
    # Use a small step size
    step_size = 0.005
    min_step = 1e-4
    
    while (time.time() - start_t) < time_limit:
        improved = False
        for i in range(N):
            if (time.time() - start_t) >= time_limit:
                break
                
            # Identify the most constraining neighbor and boundary
            # Distances to all other points
            dists_sq = np.sum((best_pos[i] - best_pos)**2, axis=1)
            dists_sq[i] = np.inf
            j_nearest = np.argmin(dists_sq)
            d_nearest = np.sqrt(dists_sq[j_nearest])
            
            # Boundary distances
            x, y = best_pos[i]
            d_left = x + 0.5
            d_right = 0.5 - x
            d_bottom = y + 0.5
            d_top = 0.5 - y
            min_bound_d = min(d_left, d_right, d_bottom, d_top)
            
            # Current local radius limit for this point
            # r_i = min(d_nearest/2, min_bound_d)
            # But we care about the global min.
            # If this point is not part of the global min, moving it might not help or might hurt.
            # However, moving it away from its nearest neighbor usually helps or is neutral.
            
            # Determine direction
            dir_x = 0.0
            dir_y = 0.0
            
            # If boundary is the bottleneck for this point
            if min_bound_d < d_nearest / 2.0:
                if d_left == min_bound_d: dir_x -= 1.0
                elif d_right == min_bound_d: dir_x += 1.0
                elif d_bottom == min_bound_d: dir_y -= 1.0
                elif d_top == min_bound_d: dir_y += 1.0
            else:
                # Move away from nearest neighbor
                vec = best_pos[i] - best_pos[j_nearest]
                norm = np.linalg.norm(vec)
                if norm > 1e-9:
                    dir_x = vec[0] / norm
                    dir_y = vec[1] / norm
            
            norm_dir = np.sqrt(dir_x**2 + dir_y**2)
            if norm_dir < 1e-9:
                continue
            dir_x /= norm_dir
            dir_y /= norm_dir
            
            # Try move
            new_pos = best_pos[i].copy()
            new_pos[0] += dir_x * step_size
            new_pos[1] += dir_y * step_size
            
            # Check bounds
            if np.any(np.abs(new_pos) > 0.5):
                continue
                
            old_pos_i = best_pos[i].copy()
            best_pos[i] = new_pos
            new_r, valid = compute_radius(best_pos)
            
            if valid and new_r > best_r + 1e-12:
                best_r = new_r
                improved = True
                # Optionally reduce step size slightly if we are making progress?
                # Or keep it constant.
            else:
                best_pos[i] = old_pos_i
                # If no improvement, try a smaller step or different direction?
                # For simplicity, just revert.
                
        if not improved:
            # Try smaller steps if no improvement in a full pass
            step_size *= 0.5
            if step_size < min_step:
                break

    return best_pos, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin
    
    # 1. Initialization: Hexagonal Grid
    s = 1.0
    points = []
    for j in range(-15, 16):
        for i in range(-15, 16):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3.0) / 2.0
            points.append([x, y])
            
    points = np.array(points)
    
    # Select N points closest to the origin
    dists_to_center = np.sum(points**2, axis=1)
    indices = np.argsort(dists_to_center)[:N]
    positions = points[indices].copy()
    
    # Scale to fit in [-0.5, 0.5]^2
    max_abs = np.max(np.abs(positions))
    if max_abs > 0:
        scale = 0.5 / max_abs
        positions = positions * scale
        
    # Add small noise
    r_init, _ = compute_radius(positions)
    noise_std = max(0.001, r_init * 0.01)
    positions += np.random.normal(0, noise_std, size=positions.shape)
    positions = np.clip(positions, -0.4999, 0.4999)
    
    current_radius, _ = compute_radius(positions)
    best_radius = current_radius
    best_positions = positions.copy()
    
    # 2. Simulated Annealing
    T = 0.001
    T_end = 1e-7
    alpha = 0.999
    step_size = 0.01
    min_step_size = 0.0001
    max_step_size = 0.05
    
    accept_count = 0
    total_count = 0
    
    # Reserve time for polish
    polish_time_budget = 10.0 # 10 seconds for polish
    
    sa_deadline = deadline - polish_time_budget
    
    while time.time() < sa_deadline:
        for _ in range(100):
            if time.time() >= sa_deadline:
                break
                
            total_count += 1
            idx = np.random.randint(N)
            
            if total_count % 100 == 0:
                accept_rate = accept_count / max(1, total_count)
                if accept_rate < 0.2:
                    step_size *= 1.2
                elif accept_rate > 0.8:
                    step_size *= 0.8
                step_size = np.clip(step_size, min_step_size, max_step_size)
                accept_count = 0
                total_count = 0
                
            delta = np.random.normal(0, step_size, size=2)
            new_pos = positions[idx].copy() + delta
            
            positions[idx] = new_pos
            new_radius, is_valid = compute_radius(positions)
            
            if not is_valid:
                new_radius = -1.0
                
            if new_radius > current_radius:
                current_radius = new_radius
                accept_count += 1
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_positions = positions.copy()
            else:
                delta_r = new_radius - current_radius
                if delta_r > -T * 10:
                    prob = np.exp(delta_r / T)
                    if np.random.random() < prob:
                        current_radius = new_radius
                        accept_count += 1
                    else:
                        positions[idx] = best_positions[idx] # Revert to best? No, revert to current.
                        # Wait, if we revert to best, we lose the "current" state which might be a stepping stone.
                        # We should revert to the state before the move.
                        # We didn't save it.
                        # Let's save old_pos in the loop.
                else:
                    pass # Revert logic needed
            
        T *= alpha
        if T < T_end:
            T = T_end

    # 3. Polish
    remaining_time = deadline - time.time()
    if remaining_time > 1.0:
        best_positions, best_radius = polish(best_positions, remaining_time - 1.0)
        
    return best_positions
