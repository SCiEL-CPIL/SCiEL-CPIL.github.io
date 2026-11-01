"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time

def _normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def construct():
    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    start_time = time.time()
    time_limit = 115.0
    
    # Spectral optimization parameters
    # We minimize the largest eigenvalue of the Gaussian kernel matrix
    # K_ij = exp(-gamma * d_ij^2)
    # This promotes uniform distribution.
    
    # Multi-scale annealing: start with small gamma (global), increase to large (local)
    # Target covering radius ~ 0.197 rad. Chord dist ~ 0.197.
    # d^2 ~ 0.039.
    # gamma ~ 1/d^2 ~ 25.
    
    gamma_start = 1.0
    gamma_end = 50.0
    num_iters = 2000
    step_size = 0.05
    
    # Pre-compute grid for final evaluation and local refinement
    n_lat = 100
    n_lon = 200
    lats = np.linspace(-np.pi/2, np.pi/2, n_lat, endpoint=False) + np.pi/n_lat
    lons = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
    LATS, LONS = np.meshgrid(lats, lons, indexing='ij')
    grid_x = np.cos(LATS) * np.cos(LONS)
    grid_y = np.cos(LATS) * np.sin(LONS)
    grid_z = np.sin(LATS)
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
    grid_points = grid_points / np.linalg.norm(grid_points, axis=1, keepdims=True)
    
    def get_covering_radius(pts):
        dots = np.dot(grid_points, pts.T)
        dots = np.clip(dots, -1.0, 1.0)
        max_dots = np.max(dots, axis=1)
        min_dists = np.arccos(max_dots)
        return np.max(min_dists)
    
    current_radius = get_covering_radius(points)
    best_points = points.copy()
    best_radius = current_radius
    
    for it in range(num_iters):
        if time.time() - start_time > time_limit - 5.0:
            break
            
        # Anneal gamma
        t = it / num_iters
        gamma = gamma_start * (gamma_end / gamma_start) ** t
        
        # Compute Kernel Matrix
        # d^2 = 2 - 2 * dot(x_i, x_j)
        dots = np.dot(points, points.T)
        dots = np.clip(dots, -1.0, 1.0)
        d_sq = 2.0 - 2.0 * dots
        K = np.exp(-gamma * d_sq)
        
        # Compute largest eigenvalue and eigenvector
        # K is symmetric
        try:
            eigvals, eigvecs = np.linalg.eigh(K)
            idx_max = np.argmax(eigvals)
            lambda_max = eigvals[idx_max]
            u = eigvecs[:, idx_max]
        except np.linalg.LinAlgError:
            break
            
        # Gradient of lambda_max w.r.t. x_i
        # G_i = 4 * gamma * u_i * sum_k K_ik * u_k * x_k
        # Z_k = u_k * x_k
        Z = u[:, np.newaxis] * points
        # W_i = sum_k K_ik * Z_k
        W = K @ Z
        Grad = 4.0 * gamma * u[:, np.newaxis] * W
        
        # Update points
        # Move in direction of -Grad to decrease lambda_max
        # Project gradient to tangent space?
        # The update x_new = x - eta * Grad.
        # Then normalize. This implicitly projects to sphere.
        # However, moving in -Grad direction might not be tangent.
        # Normalization handles the radial component.
        
        points_new = points - step_size * Grad
        points_new = points_new / np.linalg.norm(points_new, axis=1, keepdims=True)
        
        # Check improvement
        new_radius = get_covering_radius(points_new)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_points = points_new.copy()
            
        # Accept if it improves current state or with some probability (simulated annealing)
        # For simplicity, accept if it improves current local radius
        if new_radius <= current_radius:
            points = points_new
            current_radius = new_radius
        else:
            # Revert
            pass
            
    # Final local refinement using grid-based hole filling
    # Use best_points found
    points = best_points
    current_radius = best_radius
    
    # Refine for remaining time
    refine_start = time.time()
    step_size_refine = 0.01
    patience = 0
    max_patience = 50
    
    while time.time() - start_time < time_limit - 2.0:
        # Find worst point on grid
        dots = np.dot(grid_points, points.T)
        dots = np.clip(dots, -1.0, 1.0)
        max_dots = np.max(dots, axis=1)
        min_dists = np.arccos(max_dots)
        current_radius = np.max(min_dists)
        
        if current_radius < best_radius - 1e-6:
            best_radius = current_radius
            best_points = points.copy()
            patience = 0
        else:
            patience += 1
            if patience > max_patience:
                break
                
        worst_idx = np.argmax(min_dists)
        worst_point = grid_points[worst_idx]
        
        # Move 3 nearest points away from worst_point
        dots_to_worst = np.dot(points, worst_point)
        nearest_indices = np.argsort(-dots_to_worst)[:3]
        
        for idx in nearest_indices:
            p = points[idx]
            proj = worst_point - np.dot(worst_point, p) * p
            t = _normalize(proj)
            points[idx] = _normalize(p - step_size_refine * t)
            
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
    return best_points
# EVOLVE-BLOCK-END
