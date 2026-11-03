import numpy as np
import time

N = 50

def compute_score(centers):
    # centers shape (N, 3)
    # Compute pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    # Compute wall distances
    # Walls are at +/- 0.5
    # dist to wall x+ = 0.5 - x, x- = x + 0.5, etc.
    # min wall dist for particle i is min(0.5 - x_i, x_i + 0.5, 0.5 - y_i, y_i + 0.5, 0.5 - z_i, z_i + 0.5)
    # Since centers are in [-0.5, 0.5], 0.5 - x >= 0 and x + 0.5 >= 0.
    # min wall dist = 0.5 - max(|x|, |y|, |z|) ? No.
    # dist to nearest wall in x is 0.5 - |x|.
    # dist to nearest wall overall is min(0.5 - |x|, 0.5 - |y|, 0.5 - |z|).
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist, min_wall_dist)

def construct():
    start_time = time.time()
    deadline = 100.0  # Safety margin
    
    best_score = -np.inf
    best_centers = None
    
    # Initial guess: Grid
    # 4x4x4 grid has 64 points. We take 50.
    # To be more symmetric, maybe center the 50 points better?
    # Let's just use the first 50 points of a 4x4x4 grid centered at origin.
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    initial_centers = grid[:N].copy()
    
    # Run local search
    # We will try multiple restarts if time permits.
    # Each restart: start from current best or random, run local search.
    
    def local_search(centers, max_iter=2000):
        centers = centers.copy()
        current_score = compute_score(centers)
        best_local_score = current_score
        best_local_centers = centers.copy()
        
        step_size = 0.01
        noise_level = 0.05
        
        for i in range(max_iter):
            # Check time
            if time.time() > deadline:
                return best_local_centers, best_local_score
            
            # Find bottleneck
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            
            min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
            min_pair_dist = dists[min_pair_idx] / 2.0
            
            wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
            min_wall_idx = np.argmin(wall_dists)
            min_wall_dist = wall_dists[min_wall_idx]
            
            if min_pair_dist < min_wall_dist:
                # Move pair apart
                i_idx, j_idx = min_pair_idx
                v = centers[i_idx] - centers[j_idx]
                norm_v = np.linalg.norm(v)
                if norm_v < 1e-9:
                    # Random perturbation
                    v = np.random.randn(3)
                    norm_v = np.linalg.norm(v)
                v = v / norm_v
                
                centers[i_idx] += v * step_size
                centers[j_idx] -= v * step_size
            else:
                # Move away from wall
                i_idx = min_wall_idx
                # Determine which wall
                # 0.5 - |x|, 0.5 - |y|, 0.5 - |z|
                # Find the min component
                abs_c = np.abs(centers[i_idx])
                min_comp = np.argmin(0.5 - abs_c)
                # Direction to move is away from 0 in that component
                # If x is positive, move +x. If negative, move -x.
                # Actually, we want to increase 0.5 - |x|.
                # Gradient of 0.5 - |x| w.r.t x is -sign(x).
                # So we want to move in direction sign(x) to increase |x|? No.
                # We want to decrease |x|? No, we want to increase distance to wall.
                # Distance is 0.5 - |x|. To increase it, we must decrease |x|.
                # So we should move towards 0.
                # Wait. The sphere is at x. Wall is at 0.5.
                # If x > 0, dist is 0.5 - x. To increase dist, decrease x.
                # If x < 0, dist is 0.5 - (-x) = 0.5 + x. To increase dist, increase x.
                # In both cases, move towards 0.
                # Direction = -sign(centers[i_idx, min_comp])
                direction = np.zeros(3)
                direction[min_comp] = -np.sign(centers[i_idx, min_comp])
                if direction[min_comp] == 0:
                    direction[min_comp] = 1.0 # If exactly 0, pick arbitrary
                    
                centers[i_idx] += direction * step_size
            
            # Clip to box
            centers = np.clip(centers, -0.5, 0.5)
            
            # Add noise to escape local minima
            if i % 10 == 0:
                centers += np.random.randn(N, 3) * (noise_level / (1 + i/100))
                centers = np.clip(centers, -0.5, 0.5)
                
            new_score = compute_score(centers)
            if new_score > best_local_score:
                best_local_score = new_score
                best_local_centers = centers.copy()
                
            # If stuck, increase noise or restart?
            # For simplicity, we just keep going.
            
        return best_local_centers, best_local_score

    # Run initial local search
    best_centers, best_score = local_search(initial_centers)
    
    # Try random restarts if time permits
    restarts = 0
    while time.time() < deadline - 10:
        restarts += 1
        # Random initialization
        # Scale random points to fit in box with some margin
        rand_centers = np.random.rand(N, 3) - 0.5
        # Ensure they are not too close initially? 
        # Just let the local search handle it.
        
        # Run shorter local search for restarts
        new_centers, new_score = local_search(rand_centers, max_iter=500)
        
        if new_score > best_score:
            best_score = new_score
            best_centers = new_centers
            
        # If we found a very good score, we might stop early or polish
        if best_score > 0.135:
             # Polish with more iterations
             final_centers, final_score = local_search(best_centers, max_iter=1000)
             if final_score > best_score:
                 best_score = final_score
                 best_centers = final_centers
             break

    return best_centers
