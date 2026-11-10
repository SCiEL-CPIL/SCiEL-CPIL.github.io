import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    # Initial configuration: Fibonacci spiral
    # This is a good deterministic start
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Function to compute energy and gradient
    def energy_and_grad(x):
        # x is (N, 3)
        # Compute pairwise distances
        # diff[i, j] = x[i] - x[j]
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        dist = np.sqrt(dist_sq)
        
        # Mask diagonal to avoid log(0)
        mask = ~np.eye(N, dtype=bool)
        
        # Energy: -sum_{i<j} log(dist_ij)
        # We sum over i<j, so we can take the upper triangle
        # dist is symmetric.
        triu_indices = np.triu_indices(N, k=1)
        d = dist[triu_indices]
        # Avoid log(0) just in case, though points should be distinct
        d = np.maximum(d, 1e-15)
        E = -np.sum(np.log(d))
        
        # Gradient: grad_i = sum_{j != i} (x_i - x_j) / dist_ij^2
        # Note: derivative of -log(dist) w.r.t x_i is - (1/dist) * (x_i - x_j)/dist = -(x_i - x_j)/dist^2
        # Wait, E = - sum log dist. dE/dx_i = - sum (1/dist) * d(dist)/dx_i
        # dist = ||x_i - x_j||. d(dist)/dx_i = (x_i - x_j)/dist.
        # So dE/dx_i = - sum (x_i - x_j) / dist^2.
        # My previous derivation: g_i = sum (x_i - x_j)/dist^2.
        # Let's check signs.
        # If x_i moves away from x_j, dist increases, log(dist) increases, E = -sum log(dist) decreases.
        # So force should be repulsive?
        # If E decreases when dist increases, then we want to increase dist.
        # Gradient of E w.r.t x_i points in direction of increasing E.
        # dE/dx_i = - sum (x_i - x_j)/dist^2.
        # To minimize E, we move x_i in direction -grad E = sum (x_i - x_j)/dist^2.
        # This is a repulsive force. Correct.
        
        # Vectorized gradient calculation
        # term_ij = (x_i - x_j) / dist^2
        # We need sum over j for each i.
        # diff is (N, N, 3). dist_sq is (N, N).
        # We want to compute G[i] = sum_j diff[i, j] / dist_sq[i, j]
        # Set diagonal to 0 to avoid division by zero
        inv_dist_sq = np.zeros_like(dist_sq)
        np.divide(1.0, dist_sq, out=inv_dist_sq, where=(dist_sq > 1e-15))
        
        # G[i, k] = sum_j diff[i, j, k] * inv_dist_sq[i, j]
        # Using einsum: 'ijk,ij->ik'
        grad = np.einsum('ijk,ij->ik', diff, inv_dist_sq)
        
        # Project gradient to tangent space: grad_t = grad - (grad . x) x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad_t = grad - dot * x
        
        return E, grad_t

    # Optimization loop
    # We'll use a simple gradient descent with line search or fixed step.
    # Given the smoothness, a decaying step size works well.
    
    best_x = x.copy()
    best_E, _ = energy_and_grad(x)
    
    # Adaptive step size
    eta = 0.01
    min_eta = 1e-6
    
    for step in range(5000):
        if time.time() > deadline:
            break
            
        E, grad = energy_and_grad(x)
        
        # Update
        x_new = x - eta * grad
        
        # Normalize
        norms = np.linalg.norm(x_new, axis=1, keepdims=True)
        x_new = x_new / norms
        
        # Check if points are too close (invalid)
        # We rely on the repulsive force to keep them apart, but let's verify.
        
        E_new, _ = energy_and_grad(x_new)
        
        if E_new < best_E:
            best_E = E_new
            best_x = x_new.copy()
            x = x_new
            # Maybe increase eta slightly if progress is good?
            # Or keep it constant.
        else:
            # If energy increased, reduce step size
            eta *= 0.9
            if eta < min_eta:
                # Try a few more steps with tiny eta, then break if no improvement
                # For simplicity, just break if eta is very small and no progress
                if step > 100:
                    break

    return best_x
