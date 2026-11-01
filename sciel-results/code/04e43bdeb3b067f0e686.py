"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

def _energy_and_grad(X):
    """Compute logarithmic energy and gradient for points on sphere.
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    grad_i E = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    # Pairwise distances
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (N, N, 3)
    dist2 = np.sum(diff * diff, axis=2)  # (N, N)
    np.fill_diagonal(dist2, 1.0)  # avoid division by zero
    
    dist = np.sqrt(dist2)
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum over i<j of log(1/dist) = -sum over i<j of log(dist)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each i, grad_i = sum_{j != i} (x_i - x_j) / dist_ij^2
    # = sum_j (x_i - x_j) / dist2_ij  (with j != i)
    # diff[i,j] = x_i - x_j
    inv_dist2 = 1.0 / dist2  # (N, N), diagonal is 1
    np.fill_diagonal(inv_dist2, 0.0)
    
    # grad[i] = sum_j inv_dist2[i,j] * diff[i,j]
    grad = np.sum(inv_dist2[:, :, np.newaxis] * diff, axis=1)  # (N, 3)
    
    return energy, grad


def _project_to_sphere(X):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms


def _riemannian_gradient_descent(X0, max_iter=2000, tol=1e-12, step_init=0.01):
    """Riemannian gradient descent with backtracking line search on sphere."""
    X = _project_to_sphere(X0.copy())
    energy, grad = _energy_and_grad(X)
    best_energy = energy
    best_X = X.copy()
    
    step = step_init
    for it in range(max_iter):
        # Project gradient to tangent space
        grad_proj = grad - np.sum(grad * X, axis=1, keepdims=True) * X
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_proj)
        if grad_norm < tol:
            break
        
        # Backtracking line search
        alpha = step
        for _ in range(30):
            # Riemannian exponential map: move along geodesic
            # For small steps: X_new = X + alpha * grad_proj, then normalize
            # More accurately: use sin/cos for exact geodesic
            step_size = alpha * grad_norm
            if step_size > 1e-8:
                # Exact geodesic step
                v = grad_proj / grad_norm
                X_new = np.cos(step_size) * X + np.sin(step_size) * v
            else:
                X_new = X + alpha * grad_proj
            X_new = _project_to_sphere(X_new)
            
            e_new, g_new = _energy_and_grad(X_new)
            if e_new < energy:
                break
            alpha *= 0.5
        
        if e_new < best_energy:
            best_energy = e_new
            best_X = X_new.copy()
        
        energy = e_new
        grad = g_new
        X = X_new
        
        # Adaptive step size
        if e_new < best_energy:
            step = min(step * 1.1, 0.1)
        else:
            step *= 0.7
    
    return best_X, best_energy


def _lbfgs_polish(X0, max_iter=5000):
    """Polish using L-BFGS-B on flattened coordinates with normalization."""
    x0 = X0.flatten().copy()
    
    def obj_and_grad(x):
        X = x.reshape(N, 3)
        X = _project_to_sphere(X)
        e, g = _energy_and_grad(X)
        # Project gradient back to ambient space for L-BFGS-B
        g_amb = g - np.sum(g * X, axis=1, keepdims=True) * X
        return e, g_amb.flatten()
    
    result = minimize(obj_and_grad, x0, method='L-BFGS-B', 
                      jac=True, options={'maxiter': max_iter, 'gtol': 1e-12, 'ftol': 1e-15})
    X = _project_to_sphere(result.x.reshape(N, 3))
    return X, result.fun


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline, 10s margin
    
    np.random.seed(42)
    
    # Generate diverse initial points
    best_X = None
    best_E = np.inf
    
    # Fibonacci spiral as initial guess
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    candidates = [X_fib]
    
    # Add random restarts
    n_restarts = 10
    for _ in range(n_restarts):
        # Random points with some structure
        theta = np.random.uniform(0, 2 * np.pi, N)
        z = np.random.uniform(-1, 1, N)
        r = np.sqrt(1.0 - z * z)
        X_rand = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
        candidates.append(X_rand)
    
    # Also try quasi-random low-discrepancy sequences
    # Use a different golden ratio angle
    for angle_mult in [1.618, 2.414, 3.14159]:
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = angle_mult * np.pi * i
        r = np.sqrt(1.0 - z * z)
        X_alt = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        candidates.append(X_alt)
    
    for X_init in candidates:
        if time.time() > deadline:
            break
        
        # Phase 1: Riemannian GD
        X1, E1 = _riemannian_gradient_descent(X_init, max_iter=1500, tol=1e-10)
        
        if E1 < best_E:
            best_E = E1
            best_X = X1.copy()
        
        if time.time() > deadline:
            break
        
        # Phase 2: L-BFGS-B polish
        X2, E2 = _lbfgs_polish(X1, max_iter=3000)
        
        if E2 < best_E:
            best_E = E2
            best_X = X2.copy()
    
    # Final polish on best result
    if best_X is not None and time.time() < deadline:
        X_final, E_final = _lbfgs_polish(best_X, max_iter=5000)
        if E_final < best_E:
            best_E = E_final
            best_X = X_final
    
    if best_X is None:
        best_X = X_fib
    
    # Verify validity
    best_X = _project_to_sphere(best_X)
    # Check for very close points
    diff = best_X[:, np.newaxis, :] - best_X[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)
    if np.min(dist) < 1e-9:
        # Fallback to Fibonacci
        best_X = X_fib
    
    return best_X
# EVOLVE-BLOCK-END

