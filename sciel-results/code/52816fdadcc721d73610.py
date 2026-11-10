import numpy as np
import time

N = 50

def compute_radius(positions):
    """
    Compute the maximum radius for the given positions.
    Returns radius. If invalid, returns a large negative value to act as penalty.
    """
    # positions shape (N, 2)
    
    # Boundary distances
    max_abs = np.max(np.abs(positions), axis=1)
    bound_dists = 0.5 - max_abs
    
    # If any point is outside, return penalty
    if np.any(bound_dists < 0):
        return -1.0 - np.max(-bound_dists)
        
    min_bound = np.min(bound_dists)
    
    # Pairwise distance constraint
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=-1)
    
    # Ignore self-distances
    np.fill_diagonal(dists_sq, np.inf)
    
    min_dist_sq = np.min(dists_sq)
    min_dist = np.sqrt(min_dist_sq)
    
    r = min(min_bound, min_dist / 2.0)
    
    return r

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
    dists_to_center = np.sum(points**2, axis=1)
    indices = np.argsort(dists_to_center)[:N]
    positions = points[indices].copy()
    
    max_abs = np.max(np.abs(positions))
    if max_abs > 0:
        scale = 0.5 / max_abs
        positions = positions * scale
        
    r_init_val = compute_radius(positions)
    noise_std = max(0.001, r_init_val * 0.01)
    positions += np.random.normal(0, noise_std, size=positions.shape)
    positions = np.clip(positions, -0.4999, 0.4999)
    
    current_radius = compute_radius(positions)
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
    
    sa_deadline = start_time + 100.0
    
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
            
            old_pos = positions[idx].copy()
            positions[idx] = new_pos
            new_radius = compute_radius(positions)
            
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
                        positions[idx] = old_pos
                else:
                    positions[idx] = old_pos
                    
        T *= alpha
        if T < T_end:
            T = T_end
            
    # 3. Polish Phase: Constraint-Based Local Search
    def polish(pos_in):
        pos = pos_in.copy()
        current_r = compute_radius(pos)
        
        # Iterate to resolve constraints
        for _ in range(1000):
            if time.time() > deadline:
                break
                
            # Compute distances and boundaries
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=-1)
            np.fill_diagonal(dists_sq, np.inf)
            dists = np.sqrt(dists_sq)
            
            max_abs = np.max(np.abs(pos), axis=1)
            bound_dists = 0.5 - max_abs
            
            r = min(np.min(bound_dists), np.min(dists) / 2.0)
            
            if r < current_r:
                # If we made it worse, revert to previous best in this polish loop?
                # For simplicity, we just break if we don't improve after some steps?
                # Actually, the moves are small, so it shouldn't degrade much.
                pass
            
            # Threshold for active constraints
            thresh = r * 1.001 + 1e-7
            
            moved = False
            
            # 1. Boundary constraints
            for i in range(N):
                if bound_dists[i] < thresh:
                    dir = -pos[i]
                    norm = np.linalg.norm(dir)
                    if norm > 0:
                        dir = dir / norm
                    else:
                        continue
                    # Move towards center
                    step = 0.0005 * (thresh - bound_dists[i])
                    pos[i] += dir * step
                    moved = True
                    
            # 2. Pairwise constraints
            # Find pairs with dist < 2*thresh
            # We can vectorize the mask, but updating is sequential
            mask = dists < 2.0 * thresh
            for i in range(N):
                for j in range(i+1, N):
                    if mask[i, j]:
                        d = dists[i, j]
                        if d < 1e-9:
                            # If coincident, push apart randomly
                            pos[i] += np.random.normal(0, 0.001, 2)
                            pos[j] -= np.random.normal(0, 0.001, 2)
                            moved = True
                            continue
                            
                        dir_ij = (pos[i] - pos[j]) / d
                        violation = 2.0 * thresh - d
                        if violation > 0:
                            step = 0.0005 * violation
                            pos[i] += dir_ij * step
                            pos[j] -= dir_ij * step
                            moved = True
                            
            # Ensure inside square (clamp)
            pos = np.clip(pos, -0.4999, 0.4999)
            
            new_r = compute_radius(pos)
            if new_r > current_r:
                current_r = new_r
            else:
                # If no improvement, we might be stuck. 
                # Try a smaller step or break?
                # Let's just continue, the moves are small.
                pass
                
        return pos, current_r

    if time.time() < deadline - 2.0:
        polished_pos, polished_r = polish(best_positions)
        if polished_r > best_radius:
            best_positions = polished_pos
            best_radius = polished_r
            
    return best_positions
