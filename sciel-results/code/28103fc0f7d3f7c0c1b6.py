import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    # Leave 10 seconds for safety
    deadline = start_time + 110.0

    # 1. Initialization: Phyllotaxis
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    indices = np.arange(N)
    r = np.sqrt(indices / N)
    theta = indices * golden_angle
    
    # Scale to fit inside
    # Max r is 1. We want to leave room for radius r_circle.
    # Initial guess for circle radius is ~ 0.05
    # So max center distance is 0.95
    x = r * 0.95 * np.cos(theta)
    y = r * 0.95 * np.sin(theta)
    centers = np.column_stack((x, y))
    
    best_centers = centers.copy()
    best_score = calculate_score(centers)
    
    step_size = 0.02
    min_step = 0.0001
    
    for step in range(20000):
        if time.time() > deadline:
            break
        
        # Calculate distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        min_pair_dist = np.min(dists)
        norms = np.linalg.norm(centers, axis=1)
        min_boundary_dist = np.min(1.0 - norms)
        
        current_score = min(min_pair_dist / 2.0, min_boundary_dist)
        
        if current_score > best_score:
            best_score = current_score
            best_centers = centers.copy()
            
        # Determine gradient using soft-min for smoothness
        grad = np.zeros_like(centers)
        
        # 1. Pair Gradient with soft-min weighting
        # Use a smooth approximation to identify which pairs are "active"
        # soft_min weight: exp(-d^2 / sigma^2) for small distances
        sigma = 0.01  # Width of the soft-min
        d_flat = dists.flatten()
        # Mask out diagonal
        d_flat = d_flat.copy()
        d_flat[np.arange(N) * N + np.arange(N)] = np.inf
        
        # Compute soft weights for all pairs
        weights = np.exp(-(d_flat**2) / (2 * sigma**2))
        weights = weights / (np.sum(weights) + 1e-10)
        
        # Convert back to 2D
        weights_2d = weights.reshape(N, N)
        np.fill_diagonal(weights_2d, 0)
        
        # Compute weighted gradient
        # For each pair (i,j), the gradient contribution to i is w_ij * (c_i - c_j) / d_ij
        # and to j is -w_ij * (c_i - c_j) / d_ij
        # We can vectorize this
        diff_3d = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_safe = np.where(dists > 1e-6, dists, 1e-6)
        # Unit vectors
        unit_vecs = diff_3d / dists_safe[:, :, np.newaxis]
        # Weighted sum
        pair_grad = np.sum(weights_2d[:, :, np.newaxis] * unit_vecs, axis=1)
        
        # 2. Boundary Gradient with soft-min weighting
        boundary_grad = np.zeros_like(centers)
        # Soft weight for boundary: exp(-(1 - |c|)^2 / sigma_b^2)
        sigma_b = 0.01
        boundary_weights = np.exp(-((1.0 - norms)**2) / (2 * sigma_b**2))
        boundary_weights = boundary_weights / (np.sum(boundary_weights) + 1e-10)
        
        for i in range(N):
            if norms[i] > 1e-6:
                boundary_grad[i] = boundary_weights[i] * (-centers[i] / norms[i])
        
        # Combine with adaptive weights based on which constraint is tighter
        pair_score = min_pair_dist / 2.0
        bound_score = min_boundary_dist
        
        # If pair distance is the bottleneck, weight pair gradient more
        if pair_score <= bound_score:
            w_pair = 1.0
            w_bound = 0.3
        else:
            w_pair = 0.3
            w_bound = 1.0
        
        grad = w_pair * pair_grad + w_bound * boundary_grad
        
        # Normalize gradient to unit length to avoid huge steps
        g_norm = np.linalg.norm(grad)
        if g_norm > 1e-6:
            grad = grad / g_norm
            
        centers += step_size * grad
        
        # Hard constraint: Keep inside
        norms_new = np.linalg.norm(centers, axis=1)
        outside = norms_new > 0.99
        if np.any(outside):
            centers[outside] = centers[outside] * (0.99 / norms_new[outside, np.newaxis])
            
        # Decay step size
        step_size = max(min_step, step_size * 0.999)
        
    # Final polish pass with smaller steps
    if time.time() < deadline - 5:
        polish_deadline = deadline - 2
        polish_step = 0.001
        for step in range(5000):
            if time.time() > polish_deadline:
                break
            
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            
            min_pair_dist = np.min(dists)
            norms = np.linalg.norm(centers, axis=1)
            min_boundary_dist = np.min(1.0 - norms)
            
            current_score = min(min_pair_dist / 2.0, min_boundary_dist)
            
            if current_score > best_score:
                best_score = current_score
                best_centers = centers.copy()
            
            # Use only the top 10 closest pairs for precision
            flat_dists = dists.flatten()
            k = 10
            part_idx = np.argpartition(flat_dists, k)[:k]
            
            pair_grad = np.zeros_like(centers)
            for idx in part_idx:
                i = int(idx // N)
                j = int(idx % N)
                if i == j: continue
                d = dists[i, j]
                if d < 1e-6: continue
                vec = (centers[i] - centers[j]) / d
                pair_grad[i] += vec
                pair_grad[j] -= vec
            
            # Boundary gradient for points very close to boundary
            boundary_grad = np.zeros_like(centers)
            close_to_boundary = norms > (1.0 - current_score - 0.001)
            for i in np.where(close_to_boundary)[0]:
                if norms[i] > 1e-6:
                    boundary_grad[i] = -centers[i] / norms[i]
            
            # Combine
            pair_score = min_pair_dist / 2.0
            bound_score = min_boundary_dist
            
            if pair_score <= bound_score:
                w_pair = 1.0
                w_bound = 0.1
            else:
                w_pair = 0.1
                w_bound = 1.0
            
            grad = w_pair * pair_grad + w_bound * boundary_grad
            
            g_norm = np.linalg.norm(grad)
            if g_norm > 1e-6:
                grad = grad / g_norm
                
            centers += polish_step * grad
            
            # Keep inside
            norms_new = np.linalg.norm(centers, axis=1)
            outside = norms_new > 0.999
            if np.any(outside):
                centers[outside] = centers[outside] * (0.999 / norms_new[outside, np.newaxis])
            
            polish_step = max(1e-5, polish_step * 0.999)
        
        # Final check
        final_score = calculate_score(centers)
        if final_score > best_score:
            best_centers = centers.copy()
            best_score = final_score
    
    return best_centers

def calculate_score(centers):
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    norms = np.linalg.norm(centers, axis=1)
    min_bound = np.min(1.0 - norms)
    
    return min(min_pair, min_bound)
