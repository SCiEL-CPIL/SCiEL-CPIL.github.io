import numpy as np
from scipy.spatial.distance import cdist
import time

N = 100

def construct():
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_points = points.copy()
    best_energy = np.inf
    
    # Optimization loop
    max_iter = 10000
    tol = 1e-12
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        # Compute distances
        dists = cdist(points, points)
        # Avoid division by zero on diagonal
        np.fill_diagonal(dists, 1.0)
        
        # Compute energy
        # Energy = sum_{i<j} log(1/d_ij) = -0.5 * sum_{i!=j} log(d_ij)
        # Using fsum for precision as per contract
        # But for gradient steps, standard sum is usually enough to guide direction.
        # We will use standard sum for speed in loop, and fsum for final evaluation if needed.
        # Actually, the contract says "Score: ... summed with math.fsum".
        # The evaluator will calculate the score. We just need to minimize it.
        # So we can use standard numpy sum for internal optimization speed.
        
        # Compute gradient
        # g_i = sum_{j != i} (x_j - x_i) / d_ij^2
        # V_ij = x_j - x_i
        # We can compute this as:
        # diff = points[None, :, :] - points[:, None, :]  # shape (N, N, 3)
        # inv_d2 = 1.0 / (dists**2)
        # inv_d2[np.diag_indices(N)] = 0.0
        # grad = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
        
        diff = points[None, :, :] - points[:, None, :]
        inv_d2 = 1.0 / (dists**2)
        np.fill_diagonal(inv_d2, 0.0)
        
        grad = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
        
        # Project gradient to tangent space
        # proj_grad = grad - (grad . x_i) x_i
        dot_products = np.sum(grad * points, axis=1, keepdims=True)
        proj_grad = grad - dot_products * points
        
        # Line search
        # Try to find alpha such that energy decreases
        # Initial step size
        alpha = 0.01
        current_energy = -0.5 * np.sum(np.log(dists[~np.eye(N, dtype=bool)]))
        
        # Check convergence
        grad_norm = np.linalg.norm(proj_grad)
        if grad_norm < tol:
            break
            
        # Backtracking line search
        for _ in range(20):
            new_points = points + alpha * proj_grad
            # Normalize
            new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
            
            # Check for collisions (min distance > 1e-9)
            new_dists = cdist(new_points, new_points)
            min_dist = np.min(new_dists[~np.eye(N, dtype=bool)])
            if min_dist < 1e-9:
                alpha *= 0.5
                continue
                
            new_energy = -0.5 * np.sum(np.log(new_dists[~np.eye(N, dtype=bool)]))
            
            if new_energy < current_energy:
                points = new_points
                current_energy = new_energy
                break
            else:
                alpha *= 0.5
                
        if alpha < 1e-10:
            break
            
    # Final normalization
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    return points
