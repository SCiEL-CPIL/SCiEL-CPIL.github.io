"""Minimal logarithmic energy of 150 points on the sphere."""
import numpy as np
import time
import math

N = 150
DEADLINE_SECONDS = 140.0  # safety margin before 150s kill

def _log_energy_and_grad(X):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    and its gradient with respect to X (tangent to sphere, i.e., projected).
    d_ij = |x_i - x_j| (chordal distance)
    grad E / grad x_i = sum_{j != i} (x_i - x_j) / d_ij^2
    (since dE/dd_ij = -1/d_ij, and d(d_ij)/d x_i = (x_i - x_j)/d_ij)
    
    Actually: E = -sum_{i<j} log(d_ij)
    dE/dx_i = -sum_{j!=i} (1/d_ij) * (x_i - x_j)/d_ij = -sum_{j!=i} (x_i - x_j)/d_ij^2
    So grad E = for each i: -sum_{j!=i} (x_i - x_j)/d_ij^2
    Which is sum_{j!=i} (x_j - x_i)/d_ij^2
    """
    # X is (N, 3)
    # Compute pairwise distances
    # d_ij^2 = |x_i - x_j|^2
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (N, N, 3)
    d2 = np.sum(diff * diff, axis=2)  # (N, N)
    np.fill_diagonal(d2, 1.0)  # avoid division by zero on diagonal
    
    d = np.sqrt(d2)  # (N, N)
    
    # Energy: sum_{i<j} -log(d_ij)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(d[iu]))
    
    # Gradient: for each i, grad_i = sum_{j != i} (x_j - x_i) / d_ij^2
    # = sum_{j != i} diff[j, i] / d2[j, i]  ... wait
    # diff[i,j] = x_i - x_j
    # grad_i = sum_{j != i} (x_j - x_i) / d_ij^2 = -sum_{j != i} diff[i,j] / d2[i,j]
    
    # Compute: grad = -sum_{j != i} diff[i,j] / d2[i,j]
    # diff is (N, N, 3), d2 is (N, N)
    # We need to divide each diff[i,j] by d2[i,j]
    inv_d2 = 1.0 / d2  # (N, N)
    # Set diagonal to 0
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad[i] = -sum_j diff[i,j] * inv_d2[i,j]
    # diff[i,j] = x_i - x_j
    # So grad = -sum_j (x_i - x_j) / d_ij^2 = sum_j (x_j - x_i) / d_ij^2
    
    # Vectorized: for each i, sum over j of diff[i,j] * inv_d2[i,j]
    # diff[i,j] is a 3-vector, inv_d2[i,j] is scalar
    # grad_unproj = -np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    
    grad_unproj = -np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    
    # Project gradient to tangent space: grad_tan = grad_unproj - (grad_unproj . x_i) x_i
    # Since X is on unit sphere, x_i . x_i = 1
    dot = np.sum(grad_unproj * X, axis=1, keepdims=True)  # (N, 1)
    grad = grad_unproj - dot * X  # (N, 3)
    
    return energy, grad


def _project_to_sphere(X):
    """Normalize rows of X to unit length."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return X / norms


def _line_search(X, grad, max_iter=50):
    """
    Riemannian line search with backtracking.
    Start with step size, reduce until sufficient decrease.
    """
    E0, _ = _log_energy_and_grad(X)
    step = 0.1
    for _ in range(max_iter):
        # Riemannian step: X_new = X - step * grad, then project
        X_new = X - step * grad
        X_new = _project_to_sphere(X_new)
        E_new, _ = _log_energy_and_grad(X_new)
        if E_new < E0:
            return X_new, E_new
        step *= 0.5
    return X, E0


def construct():
    start_time = time.time()
    
    # Initial points: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    X = _project_to_sphere(X)
    
    E_best, _ = _log_energy_and_grad(X)
    X_best = X.copy()
    
    # Riemannian conjugate gradient
    E_cur, grad = _log_energy_and_grad(X)
    
    # R-CG parameters
    beta = 0.0
    p = -grad.copy()  # initial search direction
    iters = 0
    max_iters = 5000
    
    while time.time() - start_time < DEADLINE_SECONDS:
        iters += 1
        
        # Line search
        X_new, E_new = _line_search(X, p)
        
        if not np.all(np.isfinite(X_new)):
            break
        
        E_new2, grad_new = _log_energy_and_grad(X_new)
        
        if E_new2 < E_cur:
            # Update
            X = X_new
            E_cur = E_new2
            if E_cur < E_best:
                E_best = E_cur
                X_best = X.copy()
            
            # Check convergence
            grad_norm = np.linalg.norm(grad_new)
            if grad_norm < 1e-10:
                break
            
            # R-CG beta update (Hestenes-Stiefel)
            if beta != 0.0:
                # beta_HS = (grad_new . (grad_new - grad_prev)) / (grad_prev . grad_prev)
                # We need grad_prev; let's use a simpler approach
                pass
            
            # Use Polak-Ribiere: beta_PR = grad_new . (grad_new - grad_prev) / (grad_prev . grad_prev)
            # For simplicity, use a fixed small beta or recompute
            # Let's just restart CG every 50 iterations to avoid issues
            if iters % 50 == 0:
                p = -grad_new.copy()
            else:
                # Simple beta: use previous grad
                # We'll store prev_grad
                if hasattr(construct, '_prev_grad'):
                    prev_grad = construct._prev_grad
                    denom = np.sum(prev_grad * prev_grad)
                    if denom > 1e-20:
                        beta = np.sum(grad_new * (grad_new - prev_grad)) / denom
                    else:
                        beta = 0.0
                    beta = max(0.0, min(beta, 1.0))
                else:
                    beta = 0.0
                construct._prev_grad = grad_new.copy()
                
                p = -grad_new + beta * p
        
        else:
            # Line search failed, restart
            p = -grad_new.copy()
            E_cur = E_new2
        
        if iters >= max_iters:
            break
    
    # Final polish: use scipy minimize with BFGS for a few iterations
    try:
        from scipy.optimize import minimize
        
        # Convert to flat vector
        x0 = X_best.flatten()
        
        def obj_flat(x):
            Xf = x.reshape(N, 3)
            Xf = _project_to_sphere(Xf)
            E, _ = _log_energy_and_grad(Xf)
            return E
        
        # Run BFGS for a limited number of iterations
        res = minimize(obj_flat, x0, method='BFGS', 
                      options={'maxiter': 200, 'gtol': 1e-8})
        
        if res.success or res.fun < E_best:
            X_res = _project_to_sphere(res.x.reshape(N, 3))
            E_res, _ = _log_energy_and_grad(X_res)
            if E_res < E_best:
                E_best = E_res
                X_best = X_res.copy()
    except Exception:
        pass
    
    # Ensure validity
    X_best = _project_to_sphere(X_best)
    
    # Check for points too close
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(X_best[i] - X_best[j])
            if d < 1e-9:
                # Nudge point i slightly
                X_best[i] += 1e-6 * np.random.randn(3)
                X_best[i] /= np.linalg.norm(X_best[i])
    
    return X_best
