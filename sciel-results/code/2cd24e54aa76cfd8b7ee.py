import numpy as np

def construct():
    N = 50
    # Initialize on the inscribed sphere of radius 0.5
    # Using Fibonacci sphere for good initial distribution
    R = 0.5
    theta = np.linspace(0, np.pi, N)
    # Actually, Fibonacci sphere is better
    # phi = arccos(1 - 2*(i+0.5)/N)
    # theta = pi * (1 + sqrt(5)) * i
    i = np.arange(N)
    phi = np.arccos(1 - 2 * (i + 0.5) / N)
    theta = np.pi * (1 + np.sqrt(5)) * i
    
    centers = np.zeros((N, 3))
    centers[:, 0] = R * np.sin(phi) * np.cos(theta)
    centers[:, 1] = R * np.sin(phi) * np.sin(theta)
    centers[:, 2] = R * np.cos(phi)
    
    # Optimization
    max_iter = 5000
    step_size = 0.01
    delta_active = 0.01
    
    for i in range(max_iter):
        # 1. Compute distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Wall distances
        abs_centers = np.abs(centers)
        max_coord = np.max(abs_centers, axis=1)
        wall_dists = 0.5 - max_coord
        
        # Current radius
        min_pair_dist = np.min(dist[~np.eye(N, dtype=bool)])
        min_wall_dist = np.min(wall_dists)
        current_r = min(min_pair_dist / 2.0, min_wall_dist)
        
        # Identify active constraints
        active_pairs = np.argwhere(dist < 2 * (current_r + delta_active))
        active_pairs = active_pairs[active_pairs[:, 0] < active_pairs[:, 1]]
        
        active_walls = np.where(wall_dists < current_r + delta_active)[0]
        
        # Compute gradient
        grad = np.zeros_like(centers)
        
        # Pair contributions (push apart)
        for i_idx, j_idx in active_pairs:
            v = centers[i_idx] - centers[j_idx]
            norm_v = np.linalg.norm(v)
            if norm_v > 1e-9:
                v_norm = v / norm_v
                grad[i_idx] += v_norm
                grad[j_idx] -= v_norm
        
        # Wall contributions (push towards body diagonals to increase min wall dist)
        # For a point on the sphere, to increase 0.5 - max(|c|), 
        # we should move towards the direction where |c_x|, |c_y|, |c_z| are equal.
        # This is the direction of the nearest (sign_x, sign_y, sign_z) * (1,1,1)/sqrt(3).
        for i_idx in active_walls:
            c = centers[i_idx]
            signs = np.sign(c)
            if np.any(signs == 0):
                signs[signs == 0] = 1
            target_dir = signs / np.sqrt(3)
            # Project target_dir onto the tangent plane of the sphere at c
            # Tangent plane is orthogonal to c
            # proj = target_dir - (target_dir . c_hat) * c_hat
            c_hat = c / np.linalg.norm(c)
            proj = target_dir - np.dot(target_dir, c_hat) * c_hat
            grad[i_idx] += proj
            
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 0:
            grad /= grad_norm
            
        # Update
        centers += step_size * grad
        
        # Project back to sphere of radius R
        norms = np.linalg.norm(centers, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1e-9
        centers = centers / norms * R
        
        # Decay step size
        step_size *= 0.999
        delta_active *= 0.999
        
    return centers
