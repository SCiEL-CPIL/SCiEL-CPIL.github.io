"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 100

# EVOLVE-BLOCK-START

def _energy_and_grad(X):
    """
    Calculate logarithmic energy and gradient for N points on the unit sphere.
    X: (N, 3) array of points.
    Returns: energy (float), grad (N, 3) array.
    """
    # Vectorized calculation
    # diff[i, j] = X[i] - X[j]
    diff = X[:, None, :] - X[None, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Prevent log(0) and division by zero
    # The problem states points closer than 1e-9 are invalid, so we should avoid this.
    # For numerical stability during optimization, clip to a small value.
    dist_sq = np.maximum(dist_sq, 1e-12)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum_{i<j} log(1/dist) = - sum_{i<j} log(dist)
    # Create upper triangular mask
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    E = -np.sum(np.log(dist[mask]))
    
    # Gradient
    # grad_i = - sum_{j != i} (x_i - x_j) / dist_ij^2
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad = - sum_j (diff[i, j] * inv_dist_sq[i, j])
    # diff is (N, N, 3), inv_dist_sq is (N, N)
    # Multiply and sum over axis 1
    grad = -np.sum(diff * inv_dist_sq[:, :, None], axis=1)
    
    return E, grad

def _local_minimize(X, max_iter=100, tol=1e-9):
    """
    Local minimization using Projected Gradient Descent with Momentum.
    """
    X = X.copy()
    v = np.zeros_like(X)
    lr = 1e-3
    momentum = 0.9
    prev_E = float('inf')
    
    for i in range(max_iter):
        E, grad = _energy_and_grad(X)
        
        # Convergence check
        if abs(prev_E - E) < tol:
            break
        prev_E = E
        
        # Project gradient to tangent plane
        # g_t = g - (g . x) x
        dot = np.sum(X * grad, axis=1, keepdims=True)
        grad_t = grad - dot * X
        
        # Momentum update
        v = momentum * v - lr * grad_t
        X_new = X + v
        
        # Normalize to sphere
        norms = np.linalg.norm(X_new, axis=1, keepdims=True)
        # Avoid division by zero if something goes wrong
        norms = np.maximum(norms, 1e-9)
        X = X_new / norms
        
    return X

def construct():
    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X_init = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin for 120s limit
    
    best_E, _ = _energy_and_grad(X_init)
    best_X = X_init.copy()
    
    # Initial local minimization
    X_local = _local_minimize(X_init)
    E_local, _ = _energy_and_grad(X_local)
    if E_local < best_E:
        best_E = E_local
        best_X = X_local.copy()
        
    # Simulated Annealing Loop
    # We will run multiple restarts if time permits
    # Each restart runs SA for a fixed duration or steps
    
    T_start = 1.0
    cooling = 0.9995
    T_floor = 1e-6
    
    while time.time() < deadline:
        # Choose start point for this SA run
        # 50% chance: Best found so far
        # 50% chance: Random or Fibonacci
        if np.random.rand() < 0.5:
            X_curr = best_X.copy()
        else:
            # Random points on sphere
            X_curr = np.random.randn(N, 3)
            X_curr = X_curr / np.linalg.norm(X_curr, axis=1, keepdims=True)
            
        # Local minimize start
        X_curr = _local_minimize(X_curr)
        E_curr, _ = _energy_and_grad(X_curr)
        
        if E_curr < best_E:
            best_E = E_curr
            best_X = X_curr.copy()
            
        T = T_start
        # Run SA for a short burst (e.g., 1 second or fixed steps)
        # Let's do fixed steps to control overhead
        steps_per_burst = 2000
        steps_done = 0
        
        while steps_done < steps_per_burst and time.time() < deadline:
            # Perturbation
            # Add Gaussian noise to tangent vectors
            noise = np.random.randn(N, 3)
            dot = np.sum(X_curr * noise, axis=1, keepdims=True)
            noise_t = noise - dot * X_curr
            X_pert = X_curr + T * noise_t
            X_pert = X_pert / np.linalg.norm(X_pert, axis=1, keepdims=True)
            
            # Local minimize perturbed
            X_new = _local_minimize(X_pert, max_iter=50) # Fewer iterations for speed in inner loop
            E_new, _ = _energy_and_grad(X_new)
            
            # Metropolis
            if E_new < E_curr:
                X_curr = X_new
                E_curr = E_new
                if E_curr < best_E:
                    best_E = E_curr
                    best_X = X_curr.copy()
            else:
                delta_E = E_new - E_curr
                # Use log for numerical stability
                if delta_E / T < np.log(np.random.rand()):
                    X_curr = X_new
                    E_curr = E_new
                    
            T *= cooling
            if T < T_floor:
                T = T_floor
            steps_done += 1
            
    # Final high-precision polish
    best_X = _local_minimize(best_X, max_iter=500, tol=1e-10)
    
    return best_X

# EVOLVE-BLOCK-END
