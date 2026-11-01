"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
import time

# EVOLVE-BLOCK-START
import numpy as np
import math
from scipy.optimize import minimize

DEADLINE_FRACTION = 0.85  # safety margin
TIME_LIMIT = 150.0


def _energy_and_grad(X):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    and its gradient w.r.t. X (shape (N,3)).
    
    |x_i - x_j|^2 = 2 - 2*x_i.x_j
    d/dx_i log(1/|x_i - x_j|) = d/dx_i (-log|x_i - x_j|) = (x_i - x_j)/|x_i - x_j|^2
    So grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    # X shape (N, 3)
    # Compute pairwise dot products
    dots = X @ X.T  # (N, N), dots[i,j] = x_i . x_j
    # |x_i - x_j|^2 = 2 - 2*dots[i,j]
    d2 = 2.0 - 2.0 * dots
    # Set diagonal to 1 to avoid division by zero (we'll zero out diagonal in grad)
    np.fill_diagonal(d2, 1.0)
    
    # Energy: sum_{i<j} -log(sqrt(d2[i,j])) = -0.5 * sum_{i<j} log(d2[i,j])
    # Use upper triangle
    iu = np.triu_indices(N, 1)
    log_d2 = np.log(d2[iu])
    E = -0.5 * math.fsum(log_d2.tolist())
    
    # Gradient: grad_i = sum_{j != i} (x_i - x_j) / d2[i,j]
    # (x_i - x_j)/d2[i,j] = x_i/d2[i,j] - x_j/d2[i,j]
    # grad = X * sum_j (1/d2[i,j]) - sum_j (x_j / d2[i,j])
    inv_d2 = 1.0 / d2  # (N, N)
    # Zero out diagonal
    np.fill_diagonal(inv_d2, 0.0)
    
    # row_sums[i] = sum_j inv_d2[i,j]
    row_sums = inv_d2.sum(axis=1)  # (N,)
    
    # grad = X * row_sums[:, None] - (X.T @ inv_d2)
    # X.T @ inv_d2 has shape (3, N), so we need to transpose back
    term2 = inv_d2 @ X  # (N, 3), term2[i] = sum_j inv_d2[i,j] * x_j
    grad = X * row_sums[:, None] - term2
    
    return E, grad


def _retract(X, step):
    """
    Riemannian retraction: move along tangent direction, then project back to sphere.
    X: (N, 3) on sphere
    step: (N, 3) tangent vector (step[i] . X[i] = 0 ideally)
    """
    # First, project step to tangent space at X
    # tangent_step = step - (step . X) X
    step_dot_X = np.sum(step * X, axis=1, keepdims=True)  # (N, 1)
    tangent_step = step - step_dot_X * X
    
    # Move
    X_new = X + tangent_step
    
    # Normalize back to unit sphere
    norms = np.linalg.norm(X_new, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    X_new = X_new / norms
    
    return X_new


def _fibonacci_sphere(n):
    """Generate n points on the sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _optimize_lbfgs(X0, max_iter=3000, tol=1e-10, deadline=None):
    """
    L-BFGS optimization on the sphere using retraction.
    """
    X = X0.copy()
    n = X.shape[0]
    
    # L-BFGS memory
    m = 20
    s_list = []  # list of step vectors (flattened)
    y_list = []  # list of gradient difference vectors (flattened)
    
    E, g = _energy_and_grad(X)
    best_E = E
    best_X = X.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Check convergence
        g_flat = g.ravel()
        if np.linalg.norm(g_flat) < tol:
            break
        
        # L-BFGS two-loop recursion
        # Flatten gradient
        g_flat = g.ravel()
        
        if len(s_list) == 0:
            # Steepest descent
            direction = -g
        else:
            # Two-loop recursion
            q = g_flat.copy()
            alphas = []
            rhos = []
            
            for i in range(len(s_list) - 1, -1, -1):
                s_i = s_list[i]
                y_i = y_list[i]
                rho_i = 1.0 / np.dot(s_i, y_i)
                rhos.append(rho_i)
                alpha_i = rho_i * np.dot(s_i, q)
                alphas.append(alpha_i)
                q = q - alpha_i * y_i
            
            # Initial Hessian approximation (identity)
            r = q
            
            for i in range(len(s_list)):
                s_i = s_list[i]
                y_i = y_list[i]
                rho_i = rhos[len(s_list) - 1 - i]
                alpha_i = alphas[len(s_list) - 1 - i]
                beta_i = rho_i * np.dot(y_i, r)
                r = r + (alpha_i - beta_i) * s_i
            
            direction = r.reshape(n, 3)
        
        # Line search: backtracking
        step_size = 1.0
        X_trial = _retract(X, step_size * direction)
        E_trial, g_trial = _energy_and_grad(X_trial)
        
        # Armijo condition
        g_flat_dir = np.dot(g_flat, direction.ravel())
        
        max_ls = 30
        for ls in range(max_ls):
            if E_trial <= E + 1e-4 * step_size * g_flat_dir:
                break
            step_size *= 0.5
            X_trial = _retract(X, step_size * direction)
            E_trial, g_trial = _energy_and_grad(X_trial)
        
        if step_size < 1e-15:
            # Can't take a step
            break
        
        s_vec = (X_trial - X).ravel()
        y_vec = (g_trial - g).ravel()
        
        if np.dot(s_vec, y_vec) > 1e-10:
            s_list.append(s_vec)
            y_list.append(y_vec)
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)
        
        X = X_trial
        E = E_trial
        g = g_trial
        
        if E < best_E:
            best_E = E
            best_X = X.copy()
    
    return best_X, best_E


def _polish_steepest_descent(X0, max_iter=5000, tol=1e-14, deadline=None):
    """
    High-precision steepest descent polish with adaptive step size.
    """
    X = X0.copy()
    n = X.shape[0]
    E, g = _energy_and_grad(X)
    best_E = E
    best_X = X.copy()
    
    step_size = 1e-3
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        g_flat = g.ravel()
        g_norm = np.linalg.norm(g_flat)
        if g_norm < tol:
            break
        
        direction = -g / g_norm  # normalized steepest descent
        
        # Backtracking line search
        E_new = None
        X_trial_final = None
        g_new = None
        
        for ls in range(30):
            X_trial = _retract(X, step_size * direction)
            E_trial, g_trial = _energy_and_grad(X_trial)
            
            if E_trial < E:
                E_new = E_trial
                g_new = g_trial
                X_trial_final = X_trial
                break
            step_size *= 0.5
        
        if E_new is None:
            # Can't improve, try smaller step or break
            step_size *= 0.1
            if step_size < 1e-20:
                break
            continue
        
        X = X_trial_final
        E = E_new
        g = g_new
        
        if E < best_E:
            best_E = E
            best_X = X.copy()
        
        # Adapt step size
        if E_new < E - 1e-15:
            step_size = min(step_size * 1.5, 1e-1)
    
    return best_X, best_E


def construct():
    global _start_time
    _start_time = time.time()
    deadline = _start_time + TIME_LIMIT * DEADLINE_FRACTION
    
    # Start from Fibonacci spiral
    pts = _fibonacci_sphere(N)
    
    # Optimize using L-BFGS
    pts, best_e = _optimize_lbfgs(pts, max_iter=3000, tol=1e-10, deadline=deadline)
    
    # Polish with steepest descent
    pts, best_e = _polish_steepest_descent(pts, max_iter=5000, tol=1e-14, deadline=deadline)
    
    # Final normalization
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    
    # Verify validity
    if not np.all(np.isfinite(pts)):
        return _fibonacci_sphere(N)
    
    # Check minimum distance
    min_d = 1e18
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < min_d:
                min_d = d
    if min_d < 1e-9:
        return _fibonacci_sphere(N)
    
    return pts
# EVOLVE-BLOCK-END
