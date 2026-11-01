import numpy as np
import time
import math

N = 100

def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_energy_and_grad(points):
    """
    Compute the logarithmic energy and its gradient.
    points: (N, 3) array of unit vectors.
    Returns: energy (float), grad (N, 3) array.
    """
    # Compute pairwise distances
    # points[:, np.newaxis, :] has shape (N, 1, 3)
    # points[np.newaxis, :, :] has shape (1, N, 3)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    
    # Avoid division by zero (diagonal)
    np.fill_diagonal(dist_sq, 1.0) # Temporarily set to 1 to avoid inf/nan, mask later
    
    dist = np.sqrt(dist_sq)
    
    # Energy: sum_{i<j} -log(d_ij)
    # We compute sum_{i!=j} -log(d_ij) / 2
    # Mask the diagonal
    mask = ~np.eye(N, dtype=bool)
    
    # Gradient for each i: sum_{j != i} (x_i - x_j) / d_ij^2
    # Note: diff[i,j] = x_i - x_j
    # grad[i] = sum_j (x_i - x_j) / d_ij^2
    # We can compute this as: sum_j (diff[i,j] / dist_sq[i,j])
    
    inv_dist_sq = np.zeros_like(dist_sq)
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    
    # grad[i] = sum_j diff[i,j] * inv_dist_sq[i,j]
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    # Energy
    # E = 0.5 * sum_{i!=j} -log(d_ij)
    # log(d_ij) = 0.5 * log(d_ij^2)
    # E = -0.5 * sum_{i!=j} 0.5 * log(d_ij^2) = -0.25 * sum_{i!=j} log(d_ij^2)
    # Let's just compute directly to avoid factor confusion.
    log_d = np.zeros_like(dist)
    log_d[mask] = np.log(dist[mask])
    energy = -0.5 * np.sum(log_d[mask])
    
    return energy, grad

def construct():
    start_time = time.time()
    timeout = 110.0  # Leave 10s margin for evaluation overhead
    
    best_energy = float('inf')
    best_points = None
    
    # Initial configuration: Fibonacci sphere
    points = _fibonacci_sphere(N)
    
    # Optimization parameters
    learning_rate = 0.01
    momentum = 0.9
    v = np.zeros_like(points)
    
    # Gradient descent loop
    # We will run this for the Fibonacci start
    for step in range(10000):
        if time.time() - start_time > timeout:
            break
            
        energy, grad = _compute_energy_and_grad(points)
        
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
            
        # Update velocity
        v = momentum * v - learning_rate * grad
        
        # Update points
        points = points + v
        
        # Normalize points to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        points = points / norms
        
        # Adaptive learning rate? 
        # If energy increases significantly, we might need to reduce LR.
        # For simplicity, we stick to fixed LR with momentum, which is robust enough for this.
        
    # Check if we have time for random restarts
    # Random restarts can help escape local minima.
    # Given N=100, local minima are a risk.
    
    while time.time() - start_time < timeout - 5.0:
        # Generate random points
        points = np.random.randn(N, 3)
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Short optimization for this random start
        v = np.zeros_like(points)
        lr = 0.01
        for step in range(2000):
            if time.time() - start_time > timeout:
                break
                
            energy, grad = _compute_energy_and_grad(points)
            
            if energy < best_energy:
                best_energy = energy
                best_points = points.copy()
                
            v = momentum * v - lr * grad
            points = points + v
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            points = points / norms
            
        # If this random start didn't improve, continue to next
        # If it did, we might want to do a longer optimization on it
        # But for now, we just keep the best found.

    return best_points
