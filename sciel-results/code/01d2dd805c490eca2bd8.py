import numpy as np
import time
import math

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 seconds margin
    
    # Initial configurations
    configs = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    configs.append(fib_points)
    
    # 2. Random points (a few)
    rng = np.random.default_rng(42)
    for _ in range(5):
        pts = rng.normal(size=(N, 3))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        configs.append(pts / norms)
        
    best_energy = np.inf
    best_points = fib_points
    
    # Precompute indices for pairs
    # We will compute gradient vectorized
    
    def compute_energy_and_grad(points):
        # points: (N, 3)
        # Compute pairwise distances
        # dist[i, j] = ||points[i] - points[j]||
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Avoid division by zero
        dist[dist < 1e-12] = 1e-12
        
        # Energy: sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
        # We need upper triangle
        iu = np.triu_indices(N, k=1)
        energy = -np.sum(np.log(dist[iu]))
        
        # Gradient: grad_i = -sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
        # Note: dist^2 is ||x_i - x_j||^2
        dist2 = dist ** 2
        dist2[dist2 < 1e-24] = 1e-24
        
        # grad[i] = -sum_{j!=i} (x_i - x_j) / dist2[i, j]
        # We can compute this as:
        # sum_j (x_i - x_j) / dist2[i, j]
        # Let's use matrix operations.
        # diff is (N, N, 3). dist2 is (N, N).
        # We want to divide each diff[i, j, :] by dist2[i, j]
        # and sum over j.
        
        # Mask diagonal to avoid self-interaction (dist2[i,i]=0)
        mask = np.ones((N, N), dtype=bool)
        np.fill_diagonal(mask, False)
        
        # weighted_diff[i, j, :] = diff[i, j, :] / dist2[i, j] if mask[i, j] else 0
        # We can do this by broadcasting
        # inv_dist2 = np.zeros((N, N))
        # np.divide(1.0, dist2, out=inv_dist2, where=mask)
        
        # Actually, simpler:
        # grad = -np.sum(diff / dist2[:, :, np.newaxis], axis=1)
        # But we must handle the diagonal.
        
        # Let's create a safe inverse distance squared
        inv_dist2 = np.zeros_like(dist2)
        np.divide(1.0, dist2, out=inv_dist2, where=mask)
        
        # Now sum diff * inv_dist2[:, :, np.newaxis] over axis 1
        # diff is (N, N, 3)
        # inv_dist2 is (N, N)
        # We want sum_j diff[i, j, :] * inv_dist2[i, j]
        
        # Multiply
        weighted_diff = diff * inv_dist2[:, :, np.newaxis]
        
        # Sum over j (axis 1)
        grad = -np.sum(weighted_diff, axis=1)
        
        return energy, grad

    def optimize(points):
        current_energy, current_grad = compute_energy_and_grad(points)
        lr = 0.01
        for k in range(2000):
            if time.time() > deadline:
                break
                
            # Update
            points_new = points + lr * current_grad
            # Normalize
            norms = np.linalg.norm(points_new, axis=1, keepdims=True)
            points_new = points_new / norms
            
            new_energy, new_grad = compute_energy_and_grad(points_new)
            
            if new_energy < current_energy:
                current_energy = new_energy
                current_grad = new_grad
                points = points_new
                # Reduce learning rate slightly
                lr *= 0.99
            else:
                lr *= 0.5
                if lr < 1e-8:
                    break
                    
        return points, current_energy

    for cfg in configs:
        if time.time() > deadline:
            break
        opt_points, opt_energy = optimize(cfg)
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_points = opt_points
            
    return best_points
