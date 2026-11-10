import numpy as np
import time

N = 150

def compute_energy(points):
    # points: (N, 3)
    # Compute pairwise distances
    # Use broadcasting for efficiency
    # diff: (N, N, 3)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    # Avoid log(0) by masking diagonal
    np.fill_diagonal(dist2, 1.0)
    # Energy is sum_{i<j} log(1/dist) = -0.5 * sum_{i!=j} log(dist)
    # log(1/d) = -log(d)
    # sum_{i<j} -log(d_ij) = -0.5 * sum_{i!=j} -log(d_ij) = 0.5 * sum_{i!=j} log(d_ij)
    # Wait.
    # E = sum_{i<j} log(1/d_ij) = sum_{i<j} -log(d_ij).
    # Sum over all i!=j is 2 * sum_{i<j}.
    # So sum_{i<j} -log(d_ij) = -0.5 * sum_{i!=j} log(d_ij).
    # Let's just compute the sum over i<j directly to be safe and clear.
    
    # For efficiency, compute full sum and divide by 2.
    # log_d = -np.log(np.sqrt(dist2)) = -0.5 * np.log(dist2)
    # E = sum_{i<j} -log(d_ij) = sum_{i<j} 0.5 * log(dist2_ij)
    # Sum_{i!=j} 0.5 * log(dist2_ij) = 2 * Sum_{i<j} 0.5 * log(dist2_ij) = Sum_{i<j} log(dist2_ij)
    # So E = 0.5 * Sum_{i!=j} log(dist2_ij) is incorrect?
    # Let's check:
    # term_ij = log(1/d_ij) = -log(d_ij).
    # Sum_{i<j} term_ij.
    # Sum_{i!=j} term_ij = 2 * Sum_{i<j} term_ij.
    # So Sum_{i<j} term_ij = 0.5 * Sum_{i!=j} term_ij.
    # term_ij = -log(sqrt(dist2)) = -0.5 log(dist2).
    # Sum_{i<j} term_ij = 0.5 * Sum_{i!=j} (-0.5 log(dist2)) = -0.25 * Sum_{i!=j} log(dist2).
    
    log_dist2 = np.log(dist2)
    np.fill_diagonal(log_dist2, 0.0)
    total_sum = np.sum(log_dist2)
    energy = -0.25 * total_sum
    return energy

def compute_gradients(points):
    # points: (N, 3)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    
    # grad_i = - sum_{j != i} (x_i - x_j) / dist2_ij
    # diff[i, j] = x_i - x_j
    # We need sum over j for each i.
    # inv_dist2 = 1 / dist2
    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)
    
    # Weighted sum of differences
    # grad_raw = sum_j inv_dist2[i,j] * diff[i,j]
    # diff is (N, N, 3), inv_dist2 is (N, N)
    # We can do: np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)
    grad_raw = np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)
    
    # The gradient of energy E = sum -log(d)
    # dE/dx_i = sum_j - (1/d_ij) * d/dx_i (d_ij)
    # d_ij = ||x_i - x_j||
    # d/dx_i d_ij = (x_i - x_j) / d_ij
    # So dE/dx_i = - sum_j (x_i - x_j) / d_ij^2 = - sum_j diff[i,j] / dist2[i,j]
    # So grad_raw calculated above is exactly the gradient in R^3.
    
    # Project to tangent space
    # g_proj = g - (g . x_i) x_i
    dot_products = np.sum(grad_raw * points, axis=1)
    grad_proj = grad_raw - dot_products[:, np.newaxis] * points
    
    return grad_proj

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_energy = float('inf')
    best_points = None
    
    # Initializations
    inits = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    inits.append(fib_points)
    
    # 2. Random uniform (a few times)
    for _ in range(3):
        pts = np.random.randn(N, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        inits.append(pts)
        
    # 3. Leja points (greedy farthest point) - might be slow for 150, but let's try a simple version
    # Actually, Fibonacci is usually better than random.
    
    for init_points in inits:
        points = init_points.copy()
        current_energy = compute_energy(points)
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_points = points.copy()
            
        if time.time() > deadline:
            break
            
        # Gradient Descent
        lr = 0.01
        for step in range(1000):
            if time.time() > deadline:
                break
                
            grads = compute_gradients(points)
            
            # Normalize gradients to control step size
            grad_norms = np.linalg.norm(grads, axis=1)
            max_norm = np.max(grad_norms)
            if max_norm > 1e-12:
                # Scale gradients to have unit norm per point, then apply lr
                # Or just use fixed lr.
                # Adaptive lr: if energy increases, reduce lr.
                new_points = points - lr * grads
                new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
                
                new_energy = compute_energy(new_points)
                
                if new_energy < current_energy:
                    points = new_points
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_points = points.copy()
                    # Decrease lr slightly if progress is good
                    lr *= 0.99
                else:
                    # Backtrack
                    lr *= 0.5
                    if lr < 1e-6:
                        break
        # End gradient descent for this init
        
    if best_points is None:
        # Fallback to Fibonacci
        return fib_points
        
    return best_points
