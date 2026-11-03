import numpy as np
from scipy.optimize import minimize
import time
import math

N = 150

def _fibonacci_sphere(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _energy_and_grad(X):
    """Compute logarithmic energy and gradient for points on unit sphere.
    
    E = sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log|xi-xj|
    dE/dxi = -sum_{j≠i} (xi-xj)/|xi-xj|^2
    """
    n = X.shape[0]
    # Compute all pairwise distances
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    
    # Avoid division by zero for i=j
    np.fill_diagonal(dist_sq, 1.0)
    
    dist = np.sqrt(dist_sq)
    
    # Energy: sum over i<j of log(1/dist) = -sum over i<j of log(dist)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: dE/dxi = -sum_{j≠i} (xi-xj)/|xi-xj|^2
    # diff[i,j] = xi - xj
    # dist_sq[i,j] = |xi-xj|^2
    # So (xi-xj)/|xi-xj|^2 = diff[i,j] / dist_sq[i,j]
    # dE/dxi = -sum_{j≠i} diff[i,j] / dist_sq[i,j]
    
    # grad[i] = -sum_{j≠i} diff[i,j] / dist_sq[i,j]
    # = -sum_{j≠i} diff[i,j] / dist_sq[i,j]
    
    # diff / dist_sq[:,:,np.newaxis] gives (n,n,3)
    with np.errstate(divide='ignore', invalid='ignore'):
        grad_term = diff / (dist_sq[:, :, np.newaxis] + 1e-30)
    np.fill_diagonal(grad_term[:, :, 0], 0)  # zero out diagonal (first component)
    np.fill_diagonal(grad_term[:, :, 1], 0)
    np.fill_diagonal(grad_term[:, :, 2], 0)
    
    grad = -np.sum(grad_term, axis=1)  # (n, 3)
    
    return energy, grad

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    # Start with Fibonacci spiral
    X = _fibonacci_sphere(N)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    
    energy, grad = _energy_and_grad(X)
    best_X = X.copy()
    best_energy = energy
    
    # Projected gradient descent
    lr = 0.01
    min_lr = 1e-10
    max_iter = 5000
    tol = 1e-12
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Check convergence
        if np.linalg.norm(grad) < tol:
            break
        
        # Take gradient step
        X_new = X - lr * grad
        
        # Normalize back to unit sphere
        norms = np.linalg.norm(X_new, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        X_new = X_new / norms
        
        # Compute new energy
        energy_new, grad_new = _energy_and_grad(X_new)
        
        if energy_new < energy:
            # Accept step
            X = X_new
            grad = grad_new
            energy = energy_new
            if energy < best_energy:
                best_energy = energy
                best_X = X.copy()
            # Maybe increase learning rate slightly
            lr = min(lr * 1.05, 0.1)
        else:
            # Reject step, decrease learning rate
            lr *= 0.5
            if lr < min_lr:
                break
        
        # Every 100 iterations, try a small random perturbation to escape local minima
        if it % 100 == 99 and it > 0:
            # Try a few random perturbations
            for _ in range(5):
                if time.time() > deadline:
                    break
                X_pert = X + 0.01 * np.random.randn(N, 3)
                X_pert = X_pert / np.linalg.norm(X_pert, axis=1, keepdims=True)
                e_pert, _ = _energy_and_grad(X_pert)
                if e_pert < energy:
                    X = X_pert
                    energy = e_pert
                    if energy < best_energy:
                        best_energy = energy
                        best_X = X.copy()
                    break
    
    # Final polish with smaller steps
    if time.time() < deadline - 5:
        X = best_X.copy()
        energy, grad = _energy_and_grad(X)
        lr = 1e-4
        for it in range(2000):
            if time.time() > deadline:
                break
            if np.linalg.norm(grad) < 1e-14:
                break
            X_new = X - lr * grad
            norms = np.linalg.norm(X_new, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            X_new = X_new / norms
            energy_new, grad_new = _energy_and_grad(X_new)
            if energy_new < energy:
                X = X_new
                grad = grad_new
                energy = energy_new
                if energy < best_energy:
                    best_energy = energy
                    best_X = X.copy()
            else:
                lr *= 0.5
                if lr < 1e-12:
                    break
    
    # Ensure output is valid
    result = best_X.copy()
    # Normalize one more time
    result = result / np.linalg.norm(result, axis=1, keepdims=True)
    # Check for very close points
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(result[i] - result[j])
            if d < 1e-9:
                # Separate them slightly
                result[i] = result[i] + 1e-8 * np.random.randn(3)
                result[i] = result[i] / np.linalg.norm(result[i])
    
    return result
