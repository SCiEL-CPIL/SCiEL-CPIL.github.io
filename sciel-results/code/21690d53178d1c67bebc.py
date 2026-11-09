import numpy as np
from scipy.optimize import minimize
import time

N = 150

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Initial: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Energy and gradient
    def energy_and_grad(flat_x):
        X = flat_x.reshape(N, 3)
        # Normalize to ensure on sphere
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / norms
        
        # Compute all pairwise distances
        # d[i,j] = |X[i] - X[j]|
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, 1.0)  # avoid log(0)
        
        # Energy = -sum_{i<j} log(d_ij)
        # Use upper triangle
        iu = np.triu_indices(N, 1)
        E = -np.sum(np.log(d[iu]))
        
        # Gradient: grad_i = -sum_{j≠i} (X_i - X_j) / d_ij^2
        # diff[i,j] = X_i - X_j
        # grad_i = -sum_j diff[i,j] / d[i,j]^2
        inv_d2 = np.zeros_like(d)
        np.fill_diagonal(inv_d2, 0.0)
        inv_d2 = 1.0 / (d * d)
        inv_d2[np.diag_indices(N)] = 0.0
        
        grad = -np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
        
        return E, grad.flatten()
    
    # Use L-BFGS-B with multiple restarts
    best_E = float('inf')
    best_x = x.copy()
    
    # First optimization from Fibonacci start
    res = minimize(energy_and_grad, x.flatten(), method='L-BFGS-B', 
                   jac=True, options={'maxiter': 5000, 'gtol': 1e-10})
    
    if res.success or res.fun < best_E:
        best_E = res.fun
        best_x = res.x.reshape(N, 3)
        best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    
    # Check time
    if time.time() > deadline - 30:
        return best_x
    
    # Try different initial configurations
    # Random starts
    rng = np.random.default_rng(42)
    for trial in range(5):
        if time.time() > deadline - 30:
            break
        
        # Random points on sphere
        X_rand = rng.standard_normal((N, 3))
        X_rand /= np.linalg.norm(X_rand, axis=1, keepdims=True)
        
        res = minimize(energy_and_grad, X_rand.flatten(), method='L-BFGS-B',
                       jac=True, options={'maxiter': 3000, 'gtol': 1e-10})
        
        if res.fun < best_E:
            best_E = res.fun
            best_x = res.x.reshape(N, 3)
            best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 10:
        res = minimize(energy_and_grad, best_x.flatten(), method='L-BFGS-B',
                       jac=True, options={'maxiter': 10000, 'gtol': 1e-12})
        if res.fun < best_E:
            best_E = res.fun
            best_x = res.x.reshape(N, 3)
            best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    
    # Final normalization
    best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    
    return best_x
