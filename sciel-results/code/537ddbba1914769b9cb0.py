import numpy as np
import time

N = 100

def construct():
    # Deadline guard
    start_time = time.time()
    max_time = 115.0  # Leave 5s margin for cleanup/return
    
    best_r = 0.0
    best_c = None
    
    # Initial guess: Grid with noise
    # 5x5x4 grid
    m_x, m_y, m_z = 5, 5, 4
    ticks_x = (np.arange(m_x) + 0.5) / m_x - 0.5
    ticks_y = (np.arange(m_y) + 0.5) / m_y - 0.5
    ticks_z = (np.arange(m_z) + 0.5) / m_z - 0.5
    
    # Create grid
    grid = np.array([(x, y, z) for z in ticks_z for y in ticks_y for x in ticks_x])
    c = grid[:N].copy()
    
    # Add small noise to break symmetry
    np.random.seed(42)
    c += np.random.normal(0, 0.01, c.shape)
    
    # Ensure within bounds initially (noise might push out)
    c = np.clip(c, -0.45, 0.45)
    
    def compute_radius(c):
        # Pairwise distances
        # c is (N, 3)
        # diff is (N, N, 3)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_pair_dist = np.sqrt(np.min(dist_sq))
        
        # Wall distances
        # dist to walls: min(0.5 - |c_k|)
        abs_c = np.abs(c)
        wall_dists = 0.5 - abs_c
        min_wall_dist = np.min(wall_dists)
        
        r = min(min_pair_dist / 2.0, min_wall_dist)
        return r, min_pair_dist, min_wall_dist, dist_sq, wall_dists

    def optimize(c, max_iters=5000, base_alpha=0.05):
        nonlocal best_r, best_c
        current_r, _, _, _, _ = compute_radius(c)
        local_best_r = current_r
        local_best_c = c.copy()
        
        alpha = base_alpha
        no_improve_count = 0
        
        for i in range(max_iters):
            if time.time() - start_time > max_time:
                break
            
            r, min_pair_dist, min_wall_dist, dist_sq, wall_dists = compute_radius(c)
            
            if r > local_best_r:
                local_best_r = r
                local_best_c = c.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 50:
                    # Shake
                    c += np.random.normal(0, 0.01, c.shape)
                    c = np.clip(c, -0.49, 0.49)
                    no_improve_count = 0
                    alpha = base_alpha
                    continue
            
            # Identify active constraints
            # Threshold for "active": within 1% of min distance
            eps = 1e-5
            threshold_pair = 2.0 * (r + eps)
            threshold_wall = r + eps
            
            G = np.zeros_like(c)
            
            # Pair constraints
            # We need indices where dist < threshold_pair
            # dist_sq is available
            active_pairs_mask = (dist_sq < threshold_pair**2) & (dist_sq > 0)
            
            # To avoid double counting and diagonal, we can just iterate over upper triangle
            # But vectorized gradient accumulation is faster.
            # Let's compute gradients for all pairs in active mask.
            # diff = c_i - c_j
            # grad_i = diff / dist
            # grad_j = -diff / dist
            
            # We have dist_sq. dist = sqrt(dist_sq).
            # We only care about active pairs.
            
            # Efficient way:
            # Get indices of active pairs
            i_idx, j_idx = np.where(active_pairs_mask)
            if len(i_idx) > 0:
                diffs = c[i_idx] - c[j_idx]
                dists = np.sqrt(dist_sq[i_idx, j_idx])
                # Avoid division by zero
                dists[dists < 1e-12] = 1e-12
                grads = diffs / dists[:, np.newaxis]
                
                # Accumulate
                np.add.at(G, i_idx, grads)
                np.add.at(G, j_idx, -grads)
            
            # Wall constraints
            active_walls_mask = (wall_dists < threshold_wall)
            # wall_dists shape (N, 3)
            # For each active wall (i, k), grad is -sign(c[i,k])
            
            # We can do this vectorized
            # G[i, k] += -sign(c[i, k]) where wall_dists[i, k] < threshold
            signs = np.sign(c)
            wall_grads = -signs * active_walls_mask
            G += wall_grads
            
            # Normalize gradient?
            # The magnitude of G can vary.
            # Let's normalize G to have max norm 1 per point?
            # Or just use a small alpha.
            
            # Update
            c += alpha * G
            
            # Project to box?
            # The gradient pushes away from walls, so it should stay inside.
            # But numerical errors or large alpha might push out.
            # Clip to valid range [-0.5 + r, 0.5 - r]?
            # We don't know r exactly for the new position yet.
            # Safe clip:
            c = np.clip(c, -0.499, 0.499)
            
            # Decay alpha
            alpha *= 0.999
        
        if local_best_r > best_r:
            best_r = local_best_r
            best_c = local_best_c
        
        return best_c

    # Run optimization
    c = optimize(c)
    
    # If time remains, try a few random restarts
    while time.time() - start_time < max_time - 10:
        # New random start
        c_rand = np.random.uniform(-0.4, 0.4, (N, 3))
        # Simple repulsion initialization to avoid overlaps
        # Run a few steps of simple repulsion
        for _ in range(100):
            diff = c_rand[:, np.newaxis, :] - c_rand[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dist_sq, np.inf)
            dist = np.sqrt(dist_sq)
            dist[dist < 1e-6] = 1e-6
            force = np.sum(diff / dist[:, :, np.newaxis], axis=1)
            c_rand += 0.01 * force
            c_rand = np.clip(c_rand, -0.45, 0.45)
            
        c_opt = optimize(c_rand, max_iters=2000, base_alpha=0.02)
        
        if time.time() - start_time > max_time:
            break

    # Final check and return
    if best_c is None:
        # Fallback to grid
        best_c = grid[:N]
        
    # Ensure finite and valid
    if not np.all(np.isfinite(best_c)):
        best_c = grid[:N]
        
    return best_c
