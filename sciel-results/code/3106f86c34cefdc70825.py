import numpy as np

def construct():
    N = 50
    best_r_global = 0.0
    best_centers_global = None

    # Define a helper to run the optimization from a given start
    def optimize(centers_init, max_iter=3000):
        centers = centers_init.copy()
        step_size = 0.05
        delta_active = 0.05
        best_r = 0.0
        best_c = centers.copy()
        
        for i in range(max_iter):
            # Compute distances
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            
            abs_centers = np.abs(centers)
            max_coord = np.max(abs_centers, axis=1)
            wall_dists = 0.5 - max_coord
            
            min_pair_dist = np.min(dist[~np.eye(N, dtype=bool)])
            min_wall_dist = np.min(wall_dists)
            current_r = min(min_pair_dist / 2.0, min_wall_dist)
            
            if current_r > best_r:
                best_r = current_r
                best_c = centers.copy()
                
            # Identify active constraints
            active_pairs = np.argwhere(dist < 2 * (current_r + delta_active))
            active_pairs = active_pairs[active_pairs[:, 0] < active_pairs[:, 1]]
            
            active_walls = np.where(wall_dists < current_r + delta_active)[0]
            
            grad = np.zeros_like(centers)
            
            # Pair contributions
            for i_idx, j_idx in active_pairs:
                v = centers[i_idx] - centers[j_idx]
                norm_v = np.linalg.norm(v)
                if norm_v > 1e-9:
                    v_norm = v / norm_v
                    grad[i_idx] += v_norm
                    grad[j_idx] -= v_norm
                    
            # Wall contributions
            for i_idx in active_walls:
                # Find the coordinate that is limiting the wall distance
                # wall_dist = 0.5 - max(|c_x|, |c_y|, |c_z|)
                # Gradient is -sign(c_k) e_k where k is argmax
                limiting_coords = np.argmax(abs_centers[i_idx])
                direction = -np.sign(centers[i_idx, limiting_coords])
                grad[i_idx, limiting_coords] += direction
                
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-9:
                grad /= grad_norm
                centers += step_size * grad
                
            # Clamp
            centers = np.clip(centers, -0.5, 0.5)
            
            # Decay
            step_size *= 0.999
            delta_active *= 0.999
            
        return best_r, best_c

    # Initialization 1: Fibonacci Sphere
    R_init = 0.25
    centers1 = np.zeros((N, 3))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(N):
        cos_theta = 1.0 - 2.0 * (i + 0.5) / N
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        phi = i * golden_angle
        centers1[i, 0] = R_init * sin_theta * np.cos(phi)
        centers1[i, 1] = R_init * sin_theta * np.sin(phi)
        centers1[i, 2] = R_init * cos_theta

    r1, c1 = optimize(centers1)
    if r1 > best_r_global:
        best_r_global = r1
        best_centers_global = c1

    # Initialization 2: Random
    np.random.seed(42)
    centers2 = np.random.uniform(-0.3, 0.3, (N, 3))
    r2, c2 = optimize(centers2)
    if r2 > best_r_global:
        best_r_global = r2
        best_centers_global = c2

    # Initialization 3: Sphere Surface (as per directive inspiration)
    # Centers on sphere of radius R_surf. 
    # To allow corners, R_surf should be large, but constrained to be inside.
    # If we put them on a sphere of radius 0.35, they are well inside.
    R_surf = 0.35
    centers3 = np.zeros((N, 3))
    for i in range(N):
        cos_theta = 1.0 - 2.0 * (i + 0.5) / N
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        phi = i * golden_angle
        centers3[i, 0] = R_surf * sin_theta * np.cos(phi)
        centers3[i, 1] = R_surf * sin_theta * np.sin(phi)
        centers3[i, 2] = R_surf * cos_theta
        
    r3, c3 = optimize(centers3)
    if r3 > best_r_global:
        best_r_global = r3
        best_centers_global = c3

    return best_centers_global
