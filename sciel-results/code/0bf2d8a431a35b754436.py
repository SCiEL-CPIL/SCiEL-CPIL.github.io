"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time


def _log_energy(points):
    """Compute log(1/d) energy for points on unit sphere."""
    n = len(points)
    # Pairwise distances
    # Use broadcasting for distance computation
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    # Avoid log(0) by setting diagonal to 1
    np.fill_diagonal(dists, 1.0)
    # Energy = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dists[iu])
    return -np.sum(log_dists)


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_uniform_sphere(n, seed=42):
    """Generate random uniform points on sphere."""
    rng = np.random.RandomState(seed)
    points = rng.randn(n, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _gradient_descent(points, lr=0.01, n_iter=5000, tol=1e-8):
    """Gradient descent on sphere for log energy minimization."""
    n = len(points)
    best_energy = _log_energy(points)
    best_points = points.copy()
    
    for it in range(n_iter):
        # Compute gradients
        # E = -sum_{i<j} log(d_ij)
        # dE/dx_i = sum_{j!=i} (x_i - x_j) / d_ij^2 * (-1) * (1/d_ij) * (x_i - x_j) / d_ij
        # Actually: d/dx_i [-log(d_ij)] = -1/d_ij * d(d_ij)/dx_i = -1/d_ij * (x_i - x_j)/d_ij = -(x_i - x_j)/d_ij^2
        # So gradient of E w.r.t. x_i = sum_{j!=i} -(x_i - x_j)/d_ij^2
        # But we need to project gradient to tangent space
        
        # Compute pairwise distances and gradients
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # n x n x 3
        dists_sq = np.sum(diff * diff, axis=-1)  # n x n
        np.fill_diagonal(dists_sq, 1.0)  # avoid division by zero
        
        # Gradient: grad_i = sum_{j!=i} (x_i - x_j) / d_ij^2
        # Note: sign - we want to minimize, so move opposite to gradient
        # grad_E_i = sum_{j!=i} -(x_i - x_j) / d_ij^2
        # So descent direction = -grad_E_i = sum_{j!=i} (x_i - x_j) / d_ij^2
        
        # Compute sum_j (x_i - x_j) / d_ij^2
        # weight_ij = 1 / d_ij^2 for i != j
        weights = np.zeros_like(dists_sq)
        mask = ~np.eye(n, dtype=bool)
        weights[mask] = 1.0 / dists_sq[mask]
        
        # grad_i = sum_j weights[i,j] * (x_i - x_j)
        # = x_i * sum_j weights[i,j] - sum_j weights[i,j] * x_j
        sum_weights = np.sum(weights, axis=1)  # n
        weighted_sum = weights @ points  # n x 3
        
        grads = points * sum_weights[:, np.newaxis] - weighted_sum  # n x 3
        
        # Project gradient to tangent space: grad_tan = grad - (grad . x_i) * x_i
        dot_products = np.sum(grads * points, axis=1, keepdims=True)
        grads_tangent = grads - dot_products * points
        
        # Update
        points_new = points + lr * grads_tangent
        points_new = _project_to_sphere(points_new)
        
        energy = _log_energy(points_new)
        if energy < best_energy:
            best_energy = energy
            best_points = points_new.copy()
        
        # Check convergence
        if it % 100 == 0:
            grad_norm = np.linalg.norm(grads_tangent)
            if grad_norm < tol:
                break
        
        points = points_new
    
    return best_points, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    # Start with Fibonacci sphere
    points = _fibonacci_sphere(N)
    current_energy = _log_energy(points)
    best_points = points.copy()
    best_energy = current_energy
    
    # Try multiple optimization strategies
    strategies = [
        ("fibonacci_gd", points.copy()),
        ("random_gd", _random_uniform_sphere(N, seed=123)),
    ]
    
    for name, init_points in strategies:
        if time.time() > deadline - 30:
            break
        
        # Gradient descent with decreasing learning rate
        lr_schedule = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
        for lr in lr_schedule:
            if time.time() > deadline - 15:
                break
            
            # Run GD for a limited number of iterations
            max_iter = 2000 if lr > 0.005 else 3000
            result_points, result_energy = _gradient_descent(
                init_points, lr=lr, n_iter=max_iter, tol=1e-9
            )
            
            if result_energy < best_energy:
                best_energy = result_energy
                best_points = result_points.copy()
            
            # Continue from this point
            init_points = result_points
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        # Very fine-tuned gradient descent
        result_points, result_energy = _gradient_descent(
            best_points, lr=1e-5, n_iter=5000, tol=1e-12
        )
        if result_energy < best_energy:
            best_energy = result_energy
            best_points = result_points.copy()
    
    # Ensure valid output
    best_points = _project_to_sphere(best_points)
    
    # Check for invalid points
    if not np.all(np.isfinite(best_points)):
        return _fibonacci_sphere(N)
    
    return best_points
# EVOLVE-BLOCK-END
