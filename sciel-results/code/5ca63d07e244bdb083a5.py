import numpy as np
import time

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_r = 0
    best_centers = None
    
    # Try k=15 and k=16
    for k in [15, 16]:
        base = 200 // k
        rem = 200 % k
        sizes = [base] * (k - rem) + [base + 1] * rem
        
        max_n = max(sizes)
        r_horiz = 1.0 / (2.0 * max_n)
        r_vert = 1.0 / ((k - 1) * np.sqrt(3) + 2.0)
        r_init = min(r_horiz, r_vert)
        
        centers = []
        for i, n in enumerate(sizes):
            y = -0.5 + r_init + i * r_init * np.sqrt(3)
            for j in range(n):
                x = r_init * (1 - n + 2 * j)
                centers.append([x, y])
        
        centers = np.array(centers)
        
        # Perturb
        rng = np.random.default_rng(42)
        centers += rng.normal(0, 0.001, centers.shape)
        
        # Local Search
        r_current = calculate_radius(centers)
        step_size = 0.005 * r_current
        
        for it in range(500):
            if time.time() > deadline - 10.0:
                break
                
            # Find closest pair or wall
            dists = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
            np.fill_diagonal(dists, np.inf)
            min_pair_dist = np.min(dists)
            min_idx = np.unravel_index(np.argmin(dists), dists.shape)
            
            wall_dists = np.min([
                centers[:, 0] + 0.5,
                0.5 - centers[:, 0],
                centers[:, 1] + 0.5,
                0.5 - centers[:, 1]
            ], axis=0)
            min_wall_dist = np.min(wall_dists)
            
            if min_pair_dist < min_wall_dist * 2: # Scale wall dist to compare with pair dist? No, wall dist is r, pair dist is 2r.
                # Compare min_pair_dist/2 and min_wall_dist
                if min_pair_dist / 2.0 < min_wall_dist:
                    i, j = min_idx
                    vec = centers[i] - centers[j]
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec /= norm
                        centers[i] += vec * step_size
                        centers[j] -= vec * step_size
                else:
                    i = np.argmin(wall_dists)
                    # Move away from wall
                    if wall_dists[i] == centers[i, 0] + 0.5:
                        centers[i, 0] += step_size
                    elif wall_dists[i] == 0.5 - centers[i, 0]:
                        centers[i, 0] -= step_size
                    elif wall_dists[i] == centers[i, 1] + 0.5:
                        centers[i, 1] += step_size
                    else:
                        centers[i, 1] -= step_size
            else:
                 # Wall is more critical? 
                 # Actually just pick the global minimum constraint
                 pass # The logic above handles it by comparing values directly if we convert to radius units.
            
            # Clamp
            centers[:, 0] = np.clip(centers[:, 0], -0.5 + r_current, 0.5 - r_current)
            centers[:, 1] = np.clip(centers[:, 1], -0.5 + r_current, 0.5 - r_current)
            
            r_new = calculate_radius(centers)
            if r_new > r_current:
                r_current = r_new
                if r_current > best_r:
                    best_r = r_current
                    best_centers = centers.copy()
            else:
                step_size *= 0.95
                if step_size < 1e-5:
                    break

    return best_centers

def calculate_radius(centers):
    dists = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    wall_dists = np.min([
        centers[:, 0] + 0.5,
        0.5 - centers[:, 0],
        centers[:, 1] + 0.5,
        0.5 - centers[:, 1]
    ], axis=0)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist, min_wall_dist)
