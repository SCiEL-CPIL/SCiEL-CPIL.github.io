import numpy as np
import time

N = 100

def compute_energy(points):
    """
    Compute the logarithmic energy: sum_{i<j} log(1 / |x_i - x_j|)
    """
    # points shape (N, 3)
    # Calculate pairwise distances
    # dist[i, j] = ||points[i] - points[j]||
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # We need sum over i < j of log(1/dist[i,j]) = - sum over i < j of log(dist[i,j])
    # Create a mask for upper triangle
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    dists = dist[mask]
    
    # Avoid log(0) just in case, though points should be distinct
    dists = np.maximum(dists, 1e-15)
    
    energy = -np.sum(np.log(dists))
    return energy

def compute_gradient(points):
    """
    Compute the gradient of the energy with respect to each point.
    E = - sum_{i<j} log(||x_i - x_j||)
    dE/dx_i = sum_{j != i} (x_j - x_i) / ||x_i - x_j||^2
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :] # x_i - x_j
    dist_sq = np.sum(diff**2, axis=2)
    
    # Avoid division by zero
    dist_sq = np.maximum(dist_sq, 1e-30)
    
    # Gradient term for pair (i, j) on x_i is (x_j - x_i) / dist_sq
    # Note: diff is x_i - x_j, so -diff is x_j - x_i
    grad_terms = -diff / dist_sq[:, :, np.newaxis]
    
    # Sum over j for each i
    # The diagonal terms (i=j) are 0 because diff is 0, but we divided by max(dist_sq, 1e-30).
    # Actually, for i=j, diff is 0, so grad_term is 0.
    grad = np.sum(grad_terms, axis=1)
    
    return grad

def project_to_tangent(points, grads):
    """
    Project gradients onto the tangent space of the sphere at each point.
    g_tan = g - (g . x) x
    """
    # points shape (N, 3), grads shape (N, 3)
    dot = np.sum(points * grads, axis=1, keepdims=True)
    g_tan = grads - dot * points
    return g_tan

def steepest_descent(points, max_iter=2000, tol=1e-12, verbose=False):
    """
    Perform steepest descent on the sphere.
    """
    points = points.copy()
    current_energy = compute_energy(points)
    
    # Initial step size
    lr = 0.01
    min_lr = 1e-10
    max_lr = 1.0
    
    for k in range(max_iter):
        grads = compute_gradient(points)
        g_tan = project_to_tangent(points, grads)
        
        # Check convergence
        grad_norm = np.linalg.norm(g_tan, axis=1)
        if np.max(grad_norm) < tol:
            if verbose: print(f"Converged at iter {k}")
            break
            
        # Backtracking line search
        # Try to find alpha such that E(x - alpha * g_tan) < E(x)
        # Simplified: try a few steps
        improved = False
        alpha = lr
        for _ in range(20):
            new_points = points - alpha * g_tan
            new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
            
            new_energy = compute_energy(new_points)
            
            if new_energy < current_energy:
                points = new_points
                current_energy = new_energy
                # Increase learning rate slightly if we succeeded
                lr = min(lr * 1.1, max_lr)
                improved = True
                break
            else:
                alpha *= 0.5
                if alpha < min_lr:
                    break
        
        if not improved:
            # If we couldn't improve even with very small step, we might be stuck or at a minimum
            # Reduce learning rate
            lr = max(lr * 0.5, min_lr)
            if lr <= min_lr:
                if verbose: print(f"Stuck at iter {k}")
                break

    return points, current_energy

def construct():
    start_time = time.time()
    deadline = 140.0 # Leave 10s margin
    
    # Initial points: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = init_points
    best_energy = compute_energy(best_points)
    
    # Run optimization
    # We can try a few restarts with noise if time permits
    
    # Restart 1: Original
    opt_points, opt_energy = steepest_descent(init_points, max_iter=1000)
    if opt_energy < best_energy:
        best_points = opt_points
        best_energy = opt_energy
        
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 20:
        return best_points
    
    # Restart 2: Add small noise
    np.random.seed(42) # Reproducibility
    noise = np.random.normal(0, 0.1, size=(N, 3))
    noisy_points = init_points + noise
    noisy_points /= np.linalg.norm(noisy_points, axis=1, keepdims=True)
    
    opt_points, opt_energy = steepest_descent(noisy_points, max_iter=1000)
    if opt_energy < best_energy:
        best_points = opt_points
        best_energy = opt_energy
        
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return best_points
        
    # Restart 3: Different noise
    noise = np.random.normal(0, 0.2, size=(N, 3))
    noisy_points = init_points + noise
    noisy_points /= np.linalg.norm(noisy_points, axis=1, keepdims=True)
    
    opt_points, opt_energy = steepest_descent(noisy_points, max_iter=1000)
    if opt_energy < best_energy:
        best_points = opt_points
        best_energy = opt_energy

    return best_points
