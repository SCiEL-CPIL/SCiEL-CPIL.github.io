"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    """
    Constructs 130 points on the unit sphere to minimize the covering radius.
    Uses gradient descent on a soft-max approximation of the covering radius.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # 1. Generate a high-resolution grid of test points on the sphere
    # Using a Fibonacci spiral for uniform distribution
    M = 20000  # Number of grid points
    i_grid = np.arange(M) + 0.5
    z_grid = 1.0 - 2.0 * i_grid / M
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * i_grid
    r_grid = np.sqrt(1.0 - z_grid * z_grid)
    grid_points = np.column_stack([r_grid * np.cos(phi_grid), 
                                   r_grid * np.sin(phi_grid), 
                                   z_grid])
    
    # 2. Initial configuration: Fibonacci spiral
    i_init = np.arange(N) + 0.5
    z_init = 1.0 - 2.0 * i_init / N
    phi_init = np.pi * (1.0 + np.sqrt(5.0)) * i_init
    r_init = np.sqrt(1.0 - z_init * z_init)
    points = np.column_stack([r_init * np.cos(phi_init), 
                              r_init * np.sin(phi_init), 
                              z_init])
    
    best_points = points.copy()
    best_radius = float('inf')
    
    # Helper to compute covering radius
    def compute_covering_radius(pts):
        dots = np.dot(grid_points, pts.T)
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        return np.max(min_dists)
    
    # Initial evaluation
    current_radius = compute_covering_radius(points)
    if current_radius < best_radius:
        best_radius = current_radius
        best_points = points.copy()
    
    # Optimization parameters
    beta = 15.0  # Sharpness of the soft-max
    alpha = 0.05  # Initial step size
    min_alpha = 0.001
    decay = 0.99
    
    # Pre-allocate one-hot matrix for efficiency? 
    # The indices change, so we build it each step.
    
    while time.time() < deadline:
        # Compute nearest neighbors and distances
        dots = np.dot(grid_points, points.T)
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        
        min_dists = np.min(dists, axis=1)
        nearest_indices = np.argmin(dists, axis=1)
        
        # Update best radius
        current_radius = np.max(min_dists)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        
        # Compute gradient
        max_d = np.max(min_dists)
        # Avoid overflow in exp
        weights = beta * np.exp(beta * (min_dists - max_d))
        
        # Calculate denominator: sqrt(1 - dot^2)
        nearest_dots = dots[np.arange(M), nearest_indices]
        denom = np.sqrt(np.maximum(1.0 - nearest_dots * nearest_dots, 1e-12))
        
        # Vector part: -grid_points / denom
        vec_part = -grid_points / denom[:, np.newaxis]
        
        # Weighted vectors
        weighted_vecs = weights[:, np.newaxis] * vec_part
        
        # Accumulate into grad using matrix multiplication with one-hot
        # one_hot[i, k] = 1 if nearest_indices[i] == k
        one_hot = np.eye(N)[nearest_indices]
        grad = one_hot.T @ weighted_vecs
        
        # Project gradient to tangent space
        proj = np.sum(grad * points, axis=1, keepdims=True)
        grad = grad - proj * points
        
        # Update points
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad = grad / grad_norm
            points = points - alpha * grad
        
        # Normalize points to sphere
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Decay step size
        alpha *= decay
        if alpha < min_alpha:
            alpha = min_alpha
            
    return best_points
# EVOLVE-BLOCK-END
