import numpy as np
import time

N = 100

def _energy_and_grad(points):
    """
    Compute logarithmic energy and Riemannian gradient.
    points: (N, 3) array of unit vectors.
    """
    # Compute pairwise distances
    # dist[i, j] = |points[i] - points[j]|
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero for i=j
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum_{i<j} log(1/dist) = - sum_{i<j} log(dist)
    # We only need upper triangle
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient calculation
    # Euclidean grad for x_i: - sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # Note: dist is |x_i - x_j|.
    # term_ij = (x_i - x_j) / dist[i, j]^2
    # grad_eucl_i = - sum_j term_ij
    
    # We can compute this efficiently using matrix operations
    # inv_dist_sq = 1 / dist^2
    inv_dist_sq = 1.0 / (dist * dist)
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad_eucl[i] = - sum_j (points[i] - points[j]) * inv_dist_sq[i, j]
    # = - (points[i] * sum_j inv_dist_sq[i, j]) + sum_j points[j] * inv_dist_sq[i, j]
    
    sum_inv = np.sum(inv_dist_sq, axis=1)
    # points @ inv_dist_sq gives sum_j points[j] * inv_dist_sq[i, j]
    # Wait, matrix multiplication: (N,3) @ (N,N) -> (N,3)
    # Result[i, k] = sum_j points[j, k] * inv_dist_sq[i, j]
    # This matches sum_j points[j] * inv_dist_sq[i, j]
    
    term2 = points @ inv_dist_sq
    term1 = points * sum_inv[:, np.newaxis]
    
    grad_eucl = -(term1 - term2)
    
    # Project to tangent space: grad_riemann = grad_eucl - (grad_eucl . x) x
    dot = np.sum(grad_eucl * points, axis=1)
    grad_riemann = grad_eucl - dot[:, np.newaxis] * points
    
    return energy, grad_riemann

def _retract(points, direction):
    """
    Retraction on the sphere: R_x(v) = (x + v) / ||x + v||
    points: (N, 3)
    direction: (N, 3)
    """
    new_points = points + direction
    norms = np.linalg.norm(new_points, axis=1, keepdims=True)
    return new_points / norms

def _transport(grad, old_points, new_points):
    """
    Approximate parallel transport of gradient from old_points to new_points.
    Simple projection: grad_new = grad - (grad . new_points) new_points
    This is a first-order approximation, sufficient for CG.
    """
    dot = np.sum(grad * new_points, axis=1)
    return grad - dot[:, np.newaxis] * new_points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Initial points: Fibonacci spiral
    def get_fibonacci_points():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = get_fibonacci_points()
    best_energy = _energy_and_grad(best_points)[0]
    
    # Riemannian Conjugate Gradient
    max_iter = 1000
    restarts = 5
    
    for r in range(restarts):
        if time.time() > deadline:
            break
            
        if r == 0:
            points = best_points.copy()
        else:
            # Random initialization with some structure
            points = np.random.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            # Maybe mix with Fibonacci?
            if r == 1:
                 points = get_fibonacci_points()
                 points += np.random.randn(N, 3) * 0.1
                 points /= np.linalg.norm(points, axis=1, keepdims=True)
            elif r == 2:
                 points = get_fibonacci_points()
                 points += np.random.randn(N, 3) * 0.5
                 points /= np.linalg.norm(points, axis=1, keepdims=True)
            elif r == 3:
                 # Antipodal pairs?
                 half = N // 2
                 p_half = np.random.randn(half, 3)
                 p_half /= np.linalg.norm(p_half, axis=1, keepdims=True)
                 points = np.vstack([p_half, -p_half])
            elif r == 4:
                 # Random
                 points = np.random.randn(N, 3)
                 points /= np.linalg.norm(points, axis=1, keepdims=True)

        energy, grad = _energy_and_grad(points)
        
        # CG variables
        d = -grad
        # Ensure d is in tangent space (it is, grad is projected)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
                
            # Line search: Armijo
            alpha = 1.0
            c1 = 1e-4
            rho = 0.5
            grad_dot_d = np.sum(grad * d)
            
            # If grad_dot_d is positive, we are moving uphill? 
            # d is search direction. If grad_dot_d > 0, we should flip d?
            # In minimization, we want grad_dot_d < 0.
            if grad_dot_d > 0:
                d = -grad
                grad_dot_d = np.sum(grad * d)
            
            for _ in range(50):
                new_points = _retract(points, alpha * d)
                new_energy, new_grad = _energy_and_grad(new_points)
                
                if new_energy <= energy + c1 * alpha * grad_dot_d:
                    break
                alpha *= rho
            
            if new_energy < energy:
                # Update
                old_points = points
                points = new_points
                energy = new_energy
                grad = new_grad
                
                # Transport gradient
                grad = _transport(grad, old_points, points)
                
                # Update direction
                new_grad_dot_d = np.sum(grad * d)
                beta = -new_grad_dot_d / grad_dot_d
                # Damped Fletcher-Reeves
                beta = max(0, min(beta, 0.5))
                
                d = -grad + beta * d
                
                # Project d to tangent space
                dot_d = np.sum(d * points, axis=1)
                d = d - dot_d[:, np.newaxis] * points
                
                grad_dot_d = np.sum(grad * d)
                
                if energy < best_energy:
                    best_energy = energy
                    best_points = points.copy()
                    
                # Convergence check
                if np.linalg.norm(grad) < 1e-8:
                    break
            else:
                # Line search failed, restart CG
                d = -grad
                break

    # Final check
    if not np.all(np.isfinite(best_points)):
        return get_fibonacci_points()
        
    # Ensure normalization
    best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
