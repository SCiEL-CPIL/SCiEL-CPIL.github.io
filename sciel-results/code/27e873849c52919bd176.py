import numpy as np
import time

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0  # seconds

    # Initial points using Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    # Grid for objective evaluation
    # Use a dense grid to approximate the covering radius
    n_grid = 500  # number of grid points per dimension
    # Generate a spherical grid
    theta = np.linspace(0, np.pi, n_grid)
    phi_grid = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta, phi_grid)
    theta_grid = theta_grid.flatten()
    phi_grid = phi_grid.flatten()
    grid = np.column_stack([
        np.sin(theta_grid) * np.cos(phi_grid),
        np.sin(theta_grid) * np.sin(phi_grid),
        np.cos(theta_grid)
    ])
    # Normalize grid points
    grid_norms = np.linalg.norm(grid, axis=1, keepdims=True)
    grid = grid / grid_norms

    # Soft-min/soft-max parameters
    beta_min = 50.0  # for soft-min over configuration points
    beta_max = 10.0  # for soft-max over grid points

    # Learning rate schedule
    lr = 0.01
    lr_decay = 0.999

    # Objective function
    def objective(points):
        # Compute squared angular distances from grid to points
        # d(y, P_i)^2 = 2 - 2 * dot(y, P_i)
        dots = np.dot(grid, points.T)  # shape (n_grid, N)
        sq_dists = 2.0 - 2.0 * dots  # shape (n_grid, N)
        
        # Soft-min over configuration points
        # soft_min(x) = -soft_max(-x)
        # soft_max(x) = (1/beta) * log(sum(exp(beta * x)))
        neg_sq_dists = -sq_dists
        max_neg = np.max(neg_sq_dists, axis=1, keepdims=True)
        exp_neg = np.exp(beta_min * (neg_sq_dists - max_neg))
        log_sum_exp = np.log(np.sum(exp_neg, axis=1)) + beta_min * max_neg.squeeze()
        soft_min_sq_dists = -log_sum_exp / beta_min  # shape (n_grid,)
        
        # Soft-max over grid points
        max_soft_min = np.max(soft_min_sq_dists, keepdims=True)
        exp_soft_min = np.exp(beta_max * (soft_min_sq_dists - max_soft_min))
        log_sum_exp_max = np.log(np.sum(exp_soft_min)) + beta_max * max_soft_min
        soft_max_sq_dists = log_sum_exp_max / beta_max
        
        return soft_max_sq_dists[0, 0]

    # Gradient function
    def gradient(points):
        dots = np.dot(grid, points.T)  # shape (n_grid, N)
        sq_dists = 2.0 - 2.0 * dots  # shape (n_grid, N)
        
        # Soft-min over configuration points
        neg_sq_dists = -sq_dists
        max_neg = np.max(neg_sq_dists, axis=1, keepdims=True)
        exp_neg = np.exp(beta_min * (neg_sq_dists - max_neg))
        softmax_neg = exp_neg / np.sum(exp_neg, axis=1, keepdims=True)  # shape (n_grid, N)
        # Gradient of soft_min with respect to sq_dists
        # d(soft_min)/d(sq_dists) = -softmax_neg
        grad_soft_min = -softmax_neg  # shape (n_grid, N)
        
        # Soft-max over grid points
        soft_min_sq_dists = -np.log(np.sum(exp_neg, axis=1)) / beta_min - max_neg.squeeze() / beta_min
        # Wait, let's recompute soft_min_sq_dists correctly
        log_sum_exp = np.log(np.sum(exp_neg, axis=1)) + beta_min * max_neg.squeeze()
        soft_min_sq_dists = -log_sum_exp / beta_min
        
        max_soft_min = np.max(soft_min_sq_dists, keepdims=True)
        exp_soft_min = np.exp(beta_max * (soft_min_sq_dists - max_soft_min))
        softmax_max = exp_soft_min / np.sum(exp_soft_min)  # shape (n_grid, 1)
        
        # Gradient of soft_max with respect to soft_min_sq_dists
        grad_soft_max = beta_max * softmax_max  # shape (n_grid, 1)
        
        # Chain rule: grad_soft_max * grad_soft_min
        # grad_soft_max is (n_grid, 1), grad_soft_min is (n_grid, N)
        # We need to compute the gradient with respect to points
        # grad_soft_min is (n_grid, N), and sq_dists = 2 - 2 * dots
        # d(sq_dists)/d(points) = -2 * grid[:, :, None] * points[None, :, :]  # Not quite, let's think
        # Actually, sq_dists[i, j] = 2 - 2 * dot(grid[i], points[j])
        # d(sq_dists[i, j])/d(points[j]) = -2 * grid[i]
        
        # So the gradient with respect to points[j] is:
        # sum_i grad_soft_max[i] * grad_soft_min[i, j] * (-2 * grid[i])
        
        # Let's compute this efficiently
        # grad_soft_max is (n_grid, 1), grad_soft_min is (n_grid, N)
        # We want a matrix of shape (N, 3)
        
        # First, compute the weighted sum over grid points
        # weights[i, j] = grad_soft_max[i] * grad_soft_min[i, j]
        weights = grad_soft_max * grad_soft_min  # shape (n_grid, N)
        
        # Now, for each point j, the gradient is:
        # -2 * sum_i weights[i, j] * grid[i]
        # This is equivalent to: -2 * grid.T @ weights
        grad_points = -2.0 * (grid.T @ weights)  # shape (N, 3)
        
        return grad_points

    # Optimization loop
    for it in range(10000):
        if time.time() > deadline - 5.0:  # safety margin
            break
        
        obj = objective(points)
        grad = gradient(points)
        
        # Update points
        points = points - lr * grad
        
        # Project back to sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Decay learning rate
        lr *= lr_decay
        
        if it % 100 == 0:
            # Compute actual covering radius for monitoring
            dots = np.dot(grid, points.T)
            sq_dists = 2.0 - 2.0 * dots
            min_sq_dists = np.min(sq_dists, axis=1)
            actual_covering_radius = np.max(min_sq_dists)
            actual_covering_radius_deg = np.degrees(np.arccos(np.clip(1 - actual_covering_radius / 2, -1, 1)))
            # print(f"Iter {it}: obj={obj:.6f}, actual={actual_covering_radius_deg:.6f}")
    
    # Final projection
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    return points
# EVOLVE-BLOCK-END
