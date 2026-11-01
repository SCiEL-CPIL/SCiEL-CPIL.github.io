import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _energy_and_grad(X):
    """
    Calculate energy and gradient for points on sphere.
    X: (N, 3) array of points.
    Returns: energy, gradient (N, 3)
    """
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff**2, axis=2)  # (N, N)
    
    np.fill_diagonal(dist_sq, 1.0)
    
    dist = np.sqrt(dist_sq)
    
    mask = ~np.eye(N, dtype=bool)
    
    # Energy: sum_{i<j} log(1/d) = - sum_{i<j} log(d)
    E = -0.5 * np.sum(np.log(dist[mask]))
    
    # Gradient of E w.r.t X_k
    # dE/dX_k = sum_{j!=k} (X_j - X_k) / ||X_k - X_j||^2
    # diff[k, j] = X_k - X_j
    # We want (X_j - X_k) which is -diff[k, j]
    grad_terms = -diff / dist_sq[..., np.newaxis]  # (N, N, 3)
    
    grad = np.sum(grad_terms, axis=1)  # (N, 3)
    
    return E, grad

def _penalty_objective(X_flat, mu=1000.0):
    X = X_flat.reshape(N, 3)
    E, grad = _energy_and_grad(X)
    
    norms = np.sum(X**2, axis=1)
    penalty = mu * np.sum((norms - 1.0)**2)
    
    pen_grad = 4.0 * mu * (norms - 1.0)[:, np.newaxis] * X
    
    total_grad = grad + pen_grad
    
    return E + penalty, total_grad.flatten()

def _riemannian_steepest_descent(X, steps=100, lr=0.05):
    current_X = X.copy()
    current_E, _ = _energy_and_grad(current_X)
    
    for _ in range(steps):
        E, grad = _energy_and_grad(current_X)
        
        # Project gradient to tangent space
        dots = np.sum(grad * current_X, axis=1, keepdims=True)
        grad_t = grad - dots * current_X
        
        # Update: x - lr * grad_t
        new_X = current_X - lr * grad_t
        norms = np.linalg.norm(new_X, axis=1, keepdims=True)
        new_X = new_X / norms
        
        new_E, _ = _energy_and_grad(new_X)
        
        if new_E < current_E:
            current_X = new_X
            current_E = new_E
        else:
            lr *= 0.5
            if lr < 1e-8:
                break
                
    return current_X

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    # 1. Initialization
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_X = X.copy()
    best_E, _ = _energy_and_grad(best_X)
    
    # 2. L-BFGS Polish 1
    if time.time() < deadline - 10:
        res = minimize(_penalty_objective, X.flatten(), method='L-BFGS-B', 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
        X_opt = res.x.reshape(N, 3)
        X_opt = X_opt / np.linalg.norm(X_opt, axis=1, keepdims=True)
        E_opt, _ = _energy_and_grad(X_opt)
        if E_opt < best_E:
            best_X = X_opt
            best_E = E_opt
    
    # 3. Simulated Annealing
    if time.time() < deadline - 20:
        X_sa = best_X.copy()
        T = 0.5
        T_min = 1e-8
        alpha = 0.999
        current_E, _ = _energy_and_grad(X_sa)
        
        while T > T_min and time.time() < deadline - 20:
            batch_size = 10
            indices = np.random.choice(N, batch_size, replace=False)
            X_new = X_sa.copy()
            
            noise = np.random.normal(0, T, (batch_size, 3))
            X_new[indices] += noise
            norms = np.linalg.norm(X_new[indices], axis=1, keepdims=True)
            X_new[indices] = X_new[indices] / norms
            
            E_new, _ = _energy_and_grad(X_new)
            
            if E_new < current_E or np.random.rand() < np.exp(-(E_new - current_E) / T):
                X_sa = X_new
                current_E = E_new
                if current_E < best_E:
                    best_X = X_sa.copy()
                    best_E = current_E
            
            T *= alpha
    
    # 4. L-BFGS Polish 2
    if time.time() < deadline - 10:
        res = minimize(_penalty_objective, best_X.flatten(), method='L-BFGS-B', 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
        X_opt = res.x.reshape(N, 3)
        X_opt = X_opt / np.linalg.norm(X_opt, axis=1, keepdims=True)
        E_opt, _ = _energy_and_grad(X_opt)
        if E_opt < best_E:
            best_X = X_opt
            best_E = E_opt
    
    # 5. Steepest Descent Polish
    if time.time() < deadline - 5:
        X_final = _riemannian_steepest_descent(best_X, steps=200, lr=0.1)
        E_final, _ = _energy_and_grad(X_final)
        if E_final < best_E:
            best_X = X_final
            best_E = E_final
            
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    
    return best_X
