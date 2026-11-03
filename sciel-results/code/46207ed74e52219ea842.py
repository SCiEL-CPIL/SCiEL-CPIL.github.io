"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _energy(X):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    # X is (N, 3)
    # Compute pairwise distances
    # Use broadcasting
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (N, N, 3)
    dist = np.sqrt(np.sum(diff * diff, axis=2))       # (N, N)
    # Avoid log(0) by masking diagonal
    np.fill_diagonal(dist, 1.0)  # temporary, we'll mask later
    log_inv = -np.log(dist)
    # Sum only upper triangle
    iu = np.triu_indices(N, k=1)
    return math.fsum(log_inv[iu].tolist())

def _gradient(X):
    """Compute gradient of logarithmic energy w.r.t. X (N,3).
    E = sum_{i<j} -log(|x_i - x_j|)
    dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (N, N, 3)
    dist2 = np.sum(diff * diff, axis=2)               # (N, N)
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist2, 1.0)
    inv_dist2 = 1.0 / dist2
    grad = np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)  # (N, 3)
    return grad

def _project_to_sphere(X):
    """Normalize each row to unit length."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return X / norms

def _lbfgs_sphere(X0, max_iter=5000, tol=1e-12):
    """
    Constrained optimization on the sphere using projected gradient / L-BFGS.
    We use a simple approach: gradient descent with line search on the sphere.
    Actually, let's use scipy's minimize with constraints or a custom projected L-BFGS.
    For simplicity and speed, we'll use a projected gradient method with backtracking line search.
    """
    from scipy.optimize import minimize
    
    # We'll use scipy's L-BFGS-B with a penalty or just projected gradient.
    # Better: use scipy.optimize.minimize with method='L-BFGS-B' won't handle sphere constraint well.
    # Let's implement a simple projected gradient with adaptive step size.
    
    X = X0.copy()
    n = X.shape[0]
    
    # Compute initial gradient
    g = _gradient(X)
    
    # Project gradient onto tangent space: g_proj = g - (g·x)x
    g_proj = g - np.sum(g * X, axis=1, keepdims=True) * X
    
    # Step size schedule
    lr = 0.01
    best_energy = _energy(X)
    
    for it in range(max_iter):
        # Check convergence
        gnorm = np.linalg.norm(g_proj)
        if gnorm < tol:
            break
        
        # Try to take a step
        X_new = X - lr * g_proj
        X_new = _project_to_sphere(X_new)
        
        # Line search: backtracking
        E_new = _energy(X_new)
        if E_new < best_energy:
            X = X_new
            best_energy = E_new
            # Increase step size slightly
            lr = min(lr * 1.05, 0.1)
        else:
            # Decrease step size
            lr *= 0.5
            if lr < 1e-10:
                break
        
        # Recompute gradient
        g = _gradient(X)
        g_proj = g - np.sum(g * X, axis=1, keepdims=True) * X
    
    return X, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    X0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    X0 = _project_to_sphere(X0)
    
    best_X = X0.copy()
    best_E = _energy(X0)
    
    # Run optimization with multiple random restarts
    np.random.seed(42)
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        if restart == 0:
            X_init = X0.copy()
        else:
            # Add random perturbation
            noise = np.random.randn(N, 3) * 0.1
            X_init = _project_to_sphere(X0 + noise)
        
        # Run projected gradient
        X_opt, E_opt = _lbfgs_sphere(X_init, max_iter=2000, tol=1e-13)
        
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt.copy()
        
        # Check time
        if time.time() > deadline - 10.0:
            break
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 5.0:
        X_polish, E_polish = _lbfgs_sphere(best_X, max_iter=1000, tol=1e-14)
        if E_polish < best_E:
            best_E = E_polish
            best_X = X_polish
    
    return best_X
# EVOLVE-BLOCK-END
