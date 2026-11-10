import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    timeout = 145.0 # Safety margin for 150s limit
    
    # Best found radius and positions
    best_r = 0.0
    best_pos = None
    
    # Helper to compute score
    def compute_score(pos):
        # pos shape (N, 3)
        # Check boundaries
        if np.any(pos < -0.5) or np.any(pos > 0.5):
            return 0.0
        # Min distance to walls
        d_walls = np.min(np.minimum(0.5 - pos, pos + 0.5), axis=1)
        min_wall_dist = np.min(d_walls)
        
        # Min distance between centers
        # Using scipy's cdist for efficiency if available, else manual
        try:
            from scipy.spatial.distance import cdist
            dists = cdist(pos, pos, metric='euclidean')
        except ImportError:
            # Fallback: manual calculation (slower but safe)
            dists = np.sqrt(np.sum((pos[:, np.newaxis, :] - pos[np.newaxis, :, :])**2, axis=2))
        
        # Exclude diagonal
        np.fill_diagonal(dists, np.inf)
        min_center_dist = np.min(dists)
        
        r = min(min_wall_dist, min_center_dist / 2.0)
        return r

    # Initialization: Jittered Grid
    m = int(np.ceil(N ** (1.0 / 3.0)))
    # 100 spheres -> 4x4x4 = 64, 5x5x5 = 125. 
    # Using 5x5x5 grid and taking first 100? Or 4x5x5 = 100.
    # Let's use a 4x5x5 grid for better aspect ratio? 
    # Actually, simple random is often better for relaxation to break symmetry.
    # Let's try a few different initializations.
    
    inits = []
    
    # 1. Random uniform
    inits.append(np.random.uniform(-0.45, 0.45, (N, 3)))
    
    # 2. Jittered 5x5x5 grid (take first 100)
    m_grid = 5
    ticks = (np.arange(m_grid) + 0.5) / m_grid - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Shuffle to break symmetry? No, keep order but jitter.
    jitter = np.random.uniform(-0.1, 0.1, (N, 3))
    inits.append(grid[:N] + jitter)
    
    # 3. Jittered 4x5x5 grid
    # 4 in x, 5 in y, 5 in z
    tx = (np.arange(4) + 0.5) / 4 - 0.5
    ty = (np.arange(5) + 0.5) / 5 - 0.5
    tz = (np.arange(5) + 0.5) / 5 - 0.5
    grid2 = np.array([(x, y, z) for z in tz for y in ty for x in tx])
    jitter2 = np.random.uniform(-0.1, 0.1, (N, 3))
    inits.append(grid2[:N] + jitter2)

    for init_pos in inits:
        if time.time() - start_time > timeout:
            break
            
        # Force Directed Relaxation
        pos = init_pos.copy()
        
        # Start with a small radius and increase
        r_target = 0.05
        # Max radius is roughly 0.111
        max_iter = 5000
        lr = 0.01 # Learning rate
        
        # Relaxation loop
        for it in range(max_iter):
            if time.time() - start_time > timeout:
                break
                
            # Compute forces
            forces = np.zeros_like(pos)
            
            # Pairwise repulsion
            # Vectorized distance calculation
            # diff[i, j] = pos[i] - pos[j]
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # Mask for pairs closer than 2 * r_target
            mask = dists < 2 * r_target
            # Force magnitude: k * (2*r_target - dist)
            # We only add to forces if mask is true
            # To avoid double counting, we can sum over j for each i
            
            # Efficient way:
            # For each i, sum (pos[i] - pos[j]) / dist * (2*r_target - dist) for all j where dist < 2*r_target
            
            # Let's use a loop for clarity and to ensure correct force direction, 
            # but N=100 is small, so O(N^2) is fine (10k ops).
            for i in range(N):
                for j in range(i + 1, N):
                    d = dists[i, j]
                    if d < 2 * r_target:
                        # Force on i is away from j
                        # Force on j is away from i
                        # F = k * (2*r_target - d) * (pos[i] - pos[j]) / d
                        factor = (2 * r_target - d) / d
                        f_vec = factor * (pos[i] - pos[j])
                        forces[i] += f_vec
                        forces[j] -= f_vec
            
            # Wall forces
            # If pos[i][k] < -0.5 + r_target, push +k
            # If pos[i][k] > 0.5 - r_target, push -k
            # Distance to wall:
            # d_left = pos[i][k] - (-0.5)
            # d_right = 0.5 - pos[i][k]
            
            for i in range(N):
                for k in range(3):
                    # Left wall
                    d_l = pos[i][k] + 0.5
                    if d_l < r_target:
                        forces[i][k] += (r_target - d_l)
                    
                    # Right wall
                    d_r = 0.5 - pos[i][k]
                    if d_r < r_target:
                        forces[i][k] -= (r_target - d_r)
            
            # Update positions
            pos += lr * forces
            
            # Clamp to box to prevent escape (forces should handle it, but just in case)
            pos = np.clip(pos, -0.5, 0.5)
            
            # Calculate current min radius
            current_r = compute_score(pos)
            
            # Adapt r_target
            if current_r > r_target + 0.001:
                r_target += 0.001
                lr *= 0.9 # Reduce learning rate slightly to stabilize
            elif current_r < r_target - 0.001:
                # Stuck or oscillating?
                # Reduce r_target to allow relaxation
                r_target -= 0.0005
                lr *= 0.5
            else:
                # Stable
                lr *= 0.95
                
            # Keep track of best
            if current_r > best_r:
                best_r = current_r
                best_pos = pos.copy()
                
            # Early stop if we reached a very good radius
            if best_r > 0.111:
                break

    return best_pos if best_pos is not None else np.random.uniform(-0.45, 0.45, (N, 3))
