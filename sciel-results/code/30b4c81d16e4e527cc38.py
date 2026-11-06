"""Minimal logarithmic energy of 150 points on the unit sphere."""
import numpy as np
import time

N = 150


def _energy_and_grad(X):
    """Compute logarithmic energy and gradient for points on unit sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient with respect to X (in ambient space, before projection).
    """
    # Pairwise distances
    # D[i,j] = |X[i] - X[j]|
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (N, N, 3)
    dist = np.sqrt(np.sum(diff * diff, axis=2))  # (N, N)
    
    # Avoid log(0) for i==j
    dist_safe = np.where(np.eye(N, dtype=bool), 1.0, dist)
    
    # Energy: sum over i<j of -log(dist)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dist_safe[iu]))
    
    # Gradient: d/dX_i sum_{j != i} -log(|X_i - X_j|)
    # = sum_{j != i} (X_i - X_j) / |X_i - X_j|^2
    # For pair (i,j): contribution to grad_i is (X_i - X_j)/dist^2, to grad_j is (X_j - X_i)/dist^2
    # i.e., grad_i += (X_i - X_j)/dist^2 for all j != i
    # This equals sum_j (X_i - X_j)/dist(i,j)^2 where dist(i,i) is ignored
    
    # Compute gradient efficiently
    # G[i] = sum_{j != i} (X_i - X_j) / dist(i,j)^2
    # Note: (X_i - X_j)/dist^2 = diff[i,j] / dist[i,j]^2
    # But diff[i,j] = X[i] - X[j]
    
    # inv_dist_sq[i,j] = 1/dist(i,j)^2 for i != j, 0 for i==j
    inv_dist_sq = np.zeros_like(dist_safe)
    mask = ~np.eye(N, dtype=bool)
    inv_dist_sq[mask] = 1.0 / (dist_safe[mask] ** 2)
    
    # grad[i] = sum_j (X[i] - X[j]) * inv_dist_sq[i,j]
    # = X[i] * sum_j inv_dist_sq[i,j] - sum_j X[j] * inv_dist_sq[i,j]
    # Let row_sum[i] = sum_j inv_dist_sq[i,j]
    # Let W = inv_dist_sq (N x N), W[i,j] = 1/dist(i,j)^2
    # grad[i] = X[i] * W[i,:].sum() - W[i,:] @ X
    # In matrix form: grad = X * W.sum(axis=1, keepdims=True) - W @ X
    W = inv_dist_sq  # (N, N)
    row_sums = W.sum(axis=1)  # (N,)
    grad = X * row_sums[:, np.newaxis] - W @ X  # (N, 3)
    
    return energy, grad


def _project(X):
    """Project points back onto unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms


def _fibonacci_sphere(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with Fibonacci spiral
    X = _fibonacci_sphere(N)
    best_X = X.copy()
    best_E, _ = _energy_and_grad(best_X)
    
    # Gradient descent with adaptive step size, projected to sphere
    lr = 0.01
    min_lr = 1e-8
    max_lr = 0.1
    
    # Use a simple line search / backtracking
    while time.time() < deadline:
        E, grad = _energy_and_grad(X)
        
        # Project gradient to tangent space
        grad_tangent = grad - np.sum(grad * X, axis=1, keepdims=True) * X
        
        # Try a step
        step = lr * grad_tangent
        X_new = _project(X + step)
        
        E_new, _ = _energy_and_grad(X_new)
        
        if E_new < E:
            X = X_new
            if E_new < best_E:
                best_E = E_new
                best_X = X.copy()
            # Try to increase learning rate slightly
            lr = min(lr * 1.05, max_lr)
        else:
            # Backtrack
            lr = lr * 0.5
            if lr < min_lr:
                # Try a smaller step with current lr
                break
        
        # Periodic check
        if int(time.time() - start_time) % 10 == 0:
            pass  # just a marker
    
    # Final polish: run more iterations with decreasing step size if time permits
    if time.time() < deadline - 5.0:
        X = best_X.copy()
        lr = 1e-4
        for _ in range(10000):
            if time.time() > deadline - 2.0:
                break
            E, grad = _energy_and_grad(X)
            grad_tangent = grad - np.sum(grad * X, axis=1, keepdims=True) * X
            X_new = _project(X + lr * grad_tangent)
            E_new, _ = _energy_and_grad(X_new)
            if E_new < E:
                X = X_new
                if E_new < best_E:
                    best_E = E_new
                    best_X = X.copy()
            else:
                lr *= 0.5
                if lr < 1e-10:
                    break
    
    # Ensure valid output
    best_X = _project(best_X)
    return best_X
