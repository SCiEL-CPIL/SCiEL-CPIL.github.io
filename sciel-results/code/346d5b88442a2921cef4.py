import numpy as np
import math
import time

N = 150

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    # Fibonacci spiral initialization
    def fib_spiral():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    def project_to_sphere(X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return X / norms
    
    def energy_and_grad(X):
        """Compute log energy and gradient. Returns (energy, grad)."""
        # Pairwise distances
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(dist_sq, 1.0)  # avoid log(0)
        dist = np.sqrt(dist_sq)
        
        # Energy: sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
        # Using upper triangle
        iu = np.triu_indices(N, k=1)
        log_dists = np.log(dist[iu])
        energy = -np.sum(log_dists)
        
        # Gradient: d/dx_i sum_{j != i} log(1/|x_i - x_j|)
        # = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
        # grad[i] = sum_{j != i} (x_i - x_j) / dist_sq[i,j]
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        # grad[i] = sum_j (x_i - x_j) * inv_dist_sq[i,j]
        # = N*x_i*sum_j inv_dist_sq[i,j] - sum_j x_j * inv_dist_sq[i,j]
        # More efficiently:
        # grad[i] = sum_j (x_i - x_j) * inv_dist_sq[i,j]
        #          = x_i * sum_j inv_dist_sq[i,j] - sum_j x_j * inv_dist_sq[i,j]
        
        # Use matrix multiply
        # sum_j x_j * inv_dist_sq[i,j] = inv_dist_sq @ X (but careful with orientation)
        # inv_dist_sq[i,j] * X[j] -> sum over j: (inv_dist_sq @ X.T).T ? No.
        # Actually: result[i] = sum_j inv_dist_sq[i,j] * X[j]
        # This is (inv_dist_sq @ X) if we think of it as matrix-vector: (A @ x)_i = sum_j A[i,j] x_j
        # So: sum_x = inv_dist_sq @ X gives shape (N, 3) where (N,3)[i] = sum_j inv_dist_sq[i,j]*X[j]
        
        sum_j = inv_dist_sq @ X  # shape (N, 3)
        row_sums = inv_dist_sq.sum(axis=1, keepdims=True)  # shape (N, 1)
        grad = X * row_sums - sum_j
        
        # Project gradient to tangent space: grad - (grad . x_i) x_i
        # For energy defined on sphere, we need tangential component
        # Actually, the gradient of the energy restricted to the sphere is the tangential part
        # grad_tangent = grad - (grad * X).sum(axis=1, keepdims=True) * X
        
        # Wait, let me reconsider. The energy E = sum_{i<j} log(1/|x_i - x_j|)
        # dE/dx_i = sum_{j != i} (x_i - x_j)/|x_i - x_j|^2 (this is the Euclidean gradient)
        # To optimize on the sphere, we use the Riemannian gradient (tangential component)
        # grad_tan[i] = grad[i] - (grad[i] * x_i) * x_i
        
        dot = (grad * X).sum(axis=1, keepdims=True)
        grad_tan = grad - dot * X
        
        return energy, grad_tan
    
    def optimize(X, max_time):
        """Riemannian gradient descent with adaptive step size."""
        X = project_to_sphere(X.copy())
        E, G = energy_and_grad(X)
        
        eta = 0.01
        best_E = E
        best_X = X.copy()
        patience = 0
        max_patience = 100
        
        while time.time() < max_time:
            E_new, _ = energy_and_grad(project_to_sphere(X - eta * G))
            X_candidate = project_to_sphere(X - eta * G)
            
            if E_new < E:
                X = X_candidate
                E = E_new
                if E < best_E:
                    best_E = E
                    best_X = X.copy()
                    patience = 0
                # Increase step slightly
                eta = min(eta * 1.05, 0.1)
            else:
                patience += 1
                eta *= 0.5
                if eta < 1e-8:
                    break
                if patience > max_patience:
                    break
        
        return best_X, best_E
    
    # Multi-restart strategy
    rng = np.random.RandomState(42)
    best_X = fib_spiral()
    best_E = energy_and_grad(best_X)[0]
    
    # Try several random perturbations of the Fibonacci spiral
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Perturb the Fibonacci points
        noise_scale = 0.1 * (restart + 1) / 10.0
        X_init = best_X + noise_scale * rng.randn(N, 3)
        X_init = project_to_sphere(X_init)
        
        # Run optimization
        X_opt, E_opt = optimize(X_init, min(deadline, time.time() + (deadline - time.time()) / (20 - restart + 1)))
        
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt.copy()
    
    # Final polish with smaller steps
    if time.time() < deadline - 2.0:
        # Run one more optimization from best
        X_polished, E_polished = optimize(best_X, deadline - 1.0)
        if E_polished < best_E:
            best_E = E_polished
            best_X = X_polished.copy()
    
    return project_to_sphere(best_X)
