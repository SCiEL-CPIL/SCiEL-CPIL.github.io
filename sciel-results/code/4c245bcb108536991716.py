import numpy as np

def construct():
    N = 50
    R_init = 0.4
    
    # Initialize with Fibonacci sphere distribution
    # This provides a nearly uniform distribution on the sphere surface
    # We choose a radius R such that the trade-off between wall distance and pair distance is optimal.
    # Approximate optimal R is ~0.4
    i = np.arange(N, dtype=float)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - (i / (N - 1.0)) * 2.0
    radius_xy = np.sqrt(1.0 - y * y)
    theta = phi * i
    x = radius_xy * np.cos(theta)
    z = radius_xy * np.sin(theta)
    centers = np.column_stack((x, y, z)) * R_init

    # Optimization
    max_iter = 20000
    step_size = 0.02
    lr_decay = 0.9998

    for it in range(max_iter):
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Wall distances
        max_coord = np.max(np.abs(centers), axis=1)
        wall_dists = 0.5 - max_coord
        
        # Current radius
        min_pair_dist = np.min(dist[~np.eye(N, dtype=bool)])
        min_wall_dist = np.min(wall_dists)
        current_r = min(min_pair_dist / 2.0, min_wall_dist)
        
        # Forces
        # Pair forces
        # W_ij = max(0, 2*current_r - dist_ij)
        W = np.maximum(0.0, 2.0 * current_r - dist)
        np.fill_diagonal(W, 0.0)
        
        # Unit vectors U_ij = diff_ij / dist_ij
        dist_safe = np.where(dist > 1e-9, dist, 1.0)
        U = diff / dist_safe[:, :, np.newaxis]
        
        # Gradient for pair i: sum_j W_ij * U_ij
        grad_pair = np.sum(W[:, :, np.newaxis] * U, axis=1)
        
        # Wall forces
        viol = np.maximum(0.0, current_r - wall_dists)
        
        limiting = np.argmax(np.abs(centers), axis=1)
        signs = np.sign(centers)
        dirs = np.zeros_like(centers)
        dirs[np.arange(N), limiting] = -signs[np.arange(N), limiting]
        grad_wall = dirs * viol[:, np.newaxis]
        
        total_grad = grad_pair + grad_wall
        
        norm = np.linalg.norm(total_grad)
        if norm > 1e-9:
            centers += step_size * (total_grad / norm)
            
        # Clamp
        centers = np.clip(centers, -0.499, 0.499)
        
        step_size *= lr_decay

    return centers
