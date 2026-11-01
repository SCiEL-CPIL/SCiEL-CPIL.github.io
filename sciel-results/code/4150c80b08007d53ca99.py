import numpy as np

def construct():
    N = 50
    # Initialize unit vectors on a sphere (Fibonacci)
    u = np.zeros((N, 3))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(N):
        cos_theta = 1.0 - 2.0 * (i + 0.5) / N
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        phi = i * golden_angle
        u[i, 0] = sin_theta * np.cos(phi)
        u[i, 1] = sin_theta * np.sin(phi)
        u[i, 2] = cos_theta
    
    # Normalize to be safe
    norms = np.linalg.norm(u, axis=1, keepdims=True)
    u = u / norms

    # Initial radius R
    R = 0.3
    best_r = 0.0
    best_R = R
    best_u = u.copy()

    max_iter = 5000
    step_size = 0.01
    delta_active = 0.05

    for i in range(max_iter):
        centers = R * u
        
        # Compute distances
        # Pair distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Wall distances
        abs_centers = np.abs(centers)
        wall_dists = 0.5 - abs_centers
        min_wall_dist = np.min(wall_dists)
        
        # Min pair distance (excluding diagonal)
        mask = ~np.eye(N, dtype=bool)
        min_pair_dist = np.min(dist[mask])
        
        current_r = min(min_pair_dist / 2.0, min_wall_dist)
        
        if current_r > best_r:
            best_r = current_r
            best_R = R
            best_u = u.copy()

        # Identify active constraints
        # Pairs: dist < 2 * (current_r + delta_active)
        active_pairs = np.argwhere(dist < 2 * (current_r + delta_active))
        active_pairs = active_pairs[active_pairs[:, 0] < active_pairs[:, 1]]
        
        # Walls: wall_dist < current_r + delta_active
        # We need to track which coordinate is limiting for each point
        # Actually, for gradient calculation, we sum gradients from all "active" wall constraints
        
        grad_R = 0.0
        grad_u = np.zeros_like(u)
        
        # Pair contributions
        # r_p = R/2 * min(dist)
        # d(r_p)/dR = dist_min / 2
        # d(r_p)/du_i = R/2 * (u_i - u_j) / dist_ij  (for active pairs)
        
        pair_active_count = len(active_pairs)
        if pair_active_count > 0:
            # We approximate the gradient by averaging over active pairs
            # This is a subgradient approach
            
            # Contribution to R
            # Sum of (dist_ij / 2) for active pairs? 
            # Actually, if we use soft-min, it's weighted. Here we just sum and normalize later or use simple average.
            # Let's use the average gradient direction for the active constraints.
            
            sum_grad_R_pair = 0.0
            sum_grad_u_pair = np.zeros_like(u)
            
            for i_idx, j_idx in active_pairs:
                d_ij = dist[i_idx, j_idx]
                if d_ij > 1e-9:
                    # dr_p/dR = d_ij / 2
                    sum_grad_R_pair += d_ij / 2.0
                    
                    # dr_p/du_i = R/2 * (u_i - u_j) / d_ij
                    # Note: u_i - u_j is vector.
                    v = u[i_idx] - u[j_idx]
                    grad_vec = (R / 2.0) * (v / d_ij)
                    sum_grad_u_pair[i_idx] += grad_vec
                    sum_grad_u_pair[j_idx] -= grad_vec
                    
            # Average over active pairs to get a stable direction
            grad_R_pair = sum_grad_R_pair / pair_active_count
            grad_u_pair = sum_grad_u_pair / pair_active_count
            
        else:
            grad_R_pair = 0.0
            grad_u_pair = np.zeros_like(u)
            
        # Wall contributions
        # r_w = 0.5 - max(R * |u_k|)
        # Let M = max_{i,k} R |u_ik|. r_w = 0.5 - M.
        # Active walls are those where R|u_ik| is close to M.
        
        max_wall_val = np.max(abs_centers) # This is R * max|u_k|
        # Active wall constraints: abs_centers[i,k] > max_wall_val - delta_active
        
        # We need to sum gradients for all active (i,k)
        # dr_w/dR = - |u_ik|
        # dr_w/du_ik = - R * sign(u_ik)
        
        active_walls = np.argwhere(abs_centers > max_wall_val - delta_active)
        
        grad_R_wall = 0.0
        grad_u_wall = np.zeros_like(u)
        active_wall_count = len(active_walls)
        
        if active_wall_count > 0:
            sum_grad_R_wall = 0.0
            sum_grad_u_wall = np.zeros_like(u)
            
            for i_idx, k_idx in active_walls:
                # abs_centers[i_idx, k_idx] = R * abs(u[i_idx, k_idx])
                abs_u_val = abs(u[i_idx, k_idx])
                sign_u_val = np.sign(u[i_idx, k_idx])
                
                # dr_w/dR = - abs_u_val
                sum_grad_R_wall += -abs_u_val
                
                # dr_w/du_ik = - R * sign(u_ik)
                # Only the k-th component of u_i changes
                grad_val = -R * sign_u_val
                sum_grad_u_wall[i_idx, k_idx] += grad_val
                
            grad_R_wall = sum_grad_R_wall / active_wall_count
            grad_u_wall = sum_grad_u_wall / active_wall_count
            
        else:
            grad_R_wall = 0.0
            grad_u_wall = np.zeros_like(u)
            
        # Combine gradients
        # We want to maximize r = min(r_p, r_w).
        # If r_p < r_w, we should prioritize pair gradient.
        # If r_w < r_p, prioritize wall gradient.
        # If equal, combine.
        
        # A simple strategy: Use the gradient of the minimum constraint.
        # Or, a weighted sum based on how close each is to the minimum.
        # Here, we just pick the dominant one or average if both are active.
        
        if min_pair_dist / 2.0 < min_wall_dist:
            # Pairs are limiting
            grad_R = grad_R_pair
            grad_u = grad_u_pair
        elif min_wall_dist < min_pair_dist / 2.0:
            # Walls are limiting
            grad_R = grad_R_wall
            grad_u = grad_u_wall
        else:
            # Both are limiting (or very close)
            grad_R = 0.5 * (grad_R_pair + grad_R_wall)
            grad_u = 0.5 * (grad_u_pair + grad_u_wall)
            
        # Project grad_u onto tangent plane
        # g_tangent = g - (g . u) u
        dot_prods = np.sum(grad_u * u, axis=1, keepdims=True)
        grad_u_tangent = grad_u - dot_prods * u
        
        # Update
        # Normalize gradients to control step size
        norm_R = np.abs(grad_R)
        norm_u = np.linalg.norm(grad_u_tangent)
        
        if norm_R > 1e-9:
            R += step_size * (grad_R / norm_R)
            
        if norm_u > 1e-9:
            u += step_size * (grad_u_tangent / norm_u)
            
        # Normalize u back to unit vectors
        u_norms = np.linalg.norm(u, axis=1, keepdims=True)
        u = u / u_norms
        
        # Clamp R
        R = np.clip(R, 0.0, 0.5)
        
        # Decay step size
        step_size *= 0.999
        delta_active *= 0.999
        
    return best_R * best_u
