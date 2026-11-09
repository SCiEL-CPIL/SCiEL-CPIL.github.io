"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # 1. Initialization: Jittered Grid
    # Use a grid that fits 50 circles roughly
    # 8x7 grid = 56 slots.
    m = 8
    n = 7
    ticks_x = (np.arange(m) + 0.5) / m - 0.5
    ticks_y = (np.arange(n) + 0.5) / n - 0.5
    grid = np.array([(x, y) for y in ticks_y for x in ticks_x])
    
    # Take first 50 points
    pos = grid[:N].copy()
    
    # Add small random jitter to break symmetry
    np.random.seed(42)
    pos += np.random.uniform(-0.01, 0.01, pos.shape)
    
    # Clamp to bounds initially
    pos = np.clip(pos, -0.5, 0.5)
    
    def calc_radius(p):
        # Calculate min radius
        # Pairwise distances
        # p is (N, 2)
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Wall distances
        # dist to wall = 0.5 - max(|x|, |y|)
        # Actually, dist to wall for circle center (x,y) is min(0.5-x, 0.5+x, 0.5-y, 0.5+y)
        # Which is 0.5 - max(|x|, |y|)
        wall_dists = 0.5 - np.max(np.abs(p), axis=1)
        min_wall_dist = np.min(wall_dists)
        
        r = min(min_pair_dist / 2.0, min_wall_dist)
        
        # Find indices for min pair and min wall
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        min_wall_idx = np.argmin(wall_dists)
        
        return r, min_pair_idx, min_wall_idx

    current_r, current_min_pair, current_min_wall = calc_radius(pos)
    best_r = current_r
    best_pos = pos.copy()
    
    # SA Parameters
    T = 0.005
    T_end = 1e-5
    steps = 150000
    alpha = (T_end / T) ** (1.0 / steps)
    
    current_pos = pos.copy()
    
    for step in range(steps):
        if time.time() > deadline:
            break
            
        # Pick random index
        i = np.random.randint(N)
        
        # Propose move
        # Move size should scale with T
        move_size = 0.02 * T / 0.005 # Normalize
        move = np.random.uniform(-move_size, move_size, 2)
        
        new_pos = current_pos.copy()
        new_pos[i] += move
        
        # Check bounds
        if np.any(new_pos[i] > 0.5) or np.any(new_pos[i] < -0.5):
            # Reject or clamp? Clamping might cause bias. Reject is safer for SA.
            # But if we are near boundary, we might get stuck.
            # Let's clamp to boundary and see if it helps, or just reject.
            # Rejecting is standard for hard constraints.
            continue
        
        # Calculate new radius
        # Check if i is involved in current minimums
        i_in_pair = (current_min_pair[0] == i or current_min_pair[1] == i)
        i_is_wall = (current_min_wall == i)
        
        if i_in_pair or i_is_wall:
            # Full recompute
            new_r, new_min_pair, new_min_wall = calc_radius(new_pos)
        else:
            # Partial recompute
            # Distances from i to others
            dists_i = np.sqrt(np.sum((new_pos[i] - current_pos)**2, axis=1))
            dists_i[i] = np.inf # Exclude self
            min_pair_dist_new = np.min(dists_i)
            
            # Wall dist for i
            wall_dist_i = 0.5 - np.max(np.abs(new_pos[i]))
            
            # The global min is min(old_global_min, new_local_min)
            # Note: old_global_min is current_r
            # But if the old min was NOT involving i, it remains valid.
            # So new_r = min(current_r, min_pair_dist_new / 2.0, wall_dist_i)
            new_r = min(current_r, min_pair_dist_new / 2.0, wall_dist_i)
            
            # Update indices if new_r is smaller than current_r
            if new_r < current_r:
                # Find which one is the new min
                if min_pair_dist_new / 2.0 < current_r and min_pair_dist_new / 2.0 <= wall_dist_i:
                    # New min is pair involving i
                    j = np.argmin(dists_i)
                    new_min_pair = (i, j) if i < j else (j, i)
                    new_min_wall = current_min_wall
                elif wall_dist_i < current_r:
                    new_min_wall = i
                    new_min_pair = current_min_pair
                else:
                    # Should not happen if new_r < current_r
                    new_min_pair = current_min_pair
                    new_min_wall = current_min_wall
            else:
                new_min_pair = current_min_pair
                new_min_wall = current_min_wall
        
        # Acceptance
        delta_r = new_r - current_r
        if delta_r > 0 or np.random.rand() < np.exp(delta_r / T):
            current_pos = new_pos
            current_r = new_r
            current_min_pair = new_min_pair
            current_min_wall = new_min_wall
            
            if current_r > best_r:
                best_r = current_r
                best_pos = current_pos.copy()
        
        T *= alpha
        
    return best_pos
# EVOLVE-BLOCK-END
