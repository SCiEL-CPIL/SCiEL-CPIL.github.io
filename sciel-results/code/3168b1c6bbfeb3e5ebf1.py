"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
import time

# EVOLVE-BLOCK-START
import numpy as np
import math
import random

DEADLINE_FRACTION = 0.85  # safety margin
TIME_LIMIT = 150.0

# Precompute upper triangle indices for efficiency
_IU = np.triu_indices(N, 1)


def _energy_and_grad(X):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    and its gradient w.r.t. X (shape (N,3)).
    """
    dots = X @ X.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    
    log_d2 = np.log(d2[_IU])
    E = -0.5 * np.sum(log_d2)
    
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    row_sums = inv_d2.sum(axis=1)
    term2 = inv_d2 @ X
    grad = X * row_sums[:, None] - term2
    
    return E, grad


def _retract(X, step):
    """Riemannian retraction with tangent projection and normalization."""
    step_dot_X = np.sum(step * X, axis=1, keepdims=True)
    tangent_step = step - step_dot_X * X
    X_new = X + tangent_step
    norms = np.linalg.norm(X_new, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X_new / norms


def _fibonacci_sphere(n, offset=0.0):
    """Generate n points on the sphere using Fibonacci spiral with optional angular offset."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng=None):
    """Generate n random points on the unit sphere."""
    if rng is None:
        rng = np.random.default_rng()
    vecs = rng.standard_normal((n, 3))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-15)


def _optimize_lbfgs(X0, max_iter=2000, tol=1e-12, deadline=None):
    """L-BFGS optimization on the sphere using retraction."""
    X = X0.copy()
    n = X.shape[0]
    
    m = 20
    s_list = []
    y_list = []
    
    E, g = _energy_and_grad(X)
    best_E = E
    best_X = X.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        g_flat = g.ravel()
        if np.linalg.norm(g_flat) < tol:
            break
        
        if len(s_list) == 0:
            direction = -g
        else:
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
            
            r = q
            
            for i in range(len(s_list)):
                s_i = s_list[i]
                y_i = y_list[i]
                rho_i = rhos[len(s_list) - 1 - i]
                alpha_i = alphas[len(s_list) - 1 - i]
                beta_i = rho_i * np.dot(y_i, r)
                r = r + (alpha_i - beta_i) * s_i
            
            direction = r.reshape(n, 3)
        
        step_size = 1.0
        X_trial = _retract(X, step_size * direction)
        E_trial, g_trial = _energy_and_grad(X_trial)
        
        g_flat_dir = np.dot(g_flat, direction.ravel())
        
        for ls in range(40):
            if E_trial <= E + 1e-4 * step_size * g_flat_dir:
                break
            step_size *= 0.5
            if step_size < 1e-18:
                break
            X_trial = _retract(X, step_size * direction)
            E_trial, g_trial = _energy_and_grad(X_trial)
        
        if step_size < 1e-18:
            break
        
        s_vec = (X_trial - X).ravel()
        y_vec = (g_trial - g).ravel()
        
        if np.dot(s_vec, y_vec) > 1e-12:
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


def _polish(X0, max_iter=10000, tol=1e-15, deadline=None):
    """High-precision steepest descent polish with adaptive step size."""
    X = X0.copy()
    n = X.shape[0]
    E, g = _energy_and_grad(X)
    best_E = E
    best_X = X.copy()
    
    step_size = 1e-4
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        g_flat = g.ravel()
        g_norm = np.linalg.norm(g_flat)
        if g_norm < tol:
            break
        
        direction = -g / g_norm
        
        E_new = None
        X_trial_final = None
        g_new = None
        
        for ls in range(40):
            X_trial = _retract(X, step_size * direction)
            E_trial, g_trial = _energy_and_grad(X_trial)
            
            if E_trial < E:
                E_new = E_trial
                g_new = g_trial
                X_trial_final = X_trial
                break
            step_size *= 0.5
        
        if E_new is None:
            step_size *= 0.1
            if step_size < 1e-25:
                break
            continue
        
        X = X_trial_final
        E = E_new
        g = g_new
        
        if E < best_E:
            best_E = E
            best_X = X.copy()
        
        if E_new < E - 1e-16:
            step_size = min(step_size * 1.3, 1e-2)
    
    return best_X, best_E


def _validate(X):
    """Check if configuration is valid."""
    if not np.all(np.isfinite(X)):
        return False
    norms = np.linalg.norm(X, axis=1)
    if np.any(norms < 1e-15):
        return False
    # Check minimum pairwise distance
    dots = X @ X.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, np.inf)
    min_d2 = np.min(d2)
    return min_d2 > 1e-18  # distance > 1e-9


def construct():
    global _start_time
    _start_time = time.time()
    deadline = _start_time + TIME_LIMIT * DEADLINE_FRACTION
    
    rng = np.random.default_rng(42)
    
    best_X = None
    best_E = float('inf')
    
    # Try multiple starting configurations
    starts = [
        _fibonacci_sphere(N, offset=0.0),
        _fibonacci_sphere(N, offset=0.3),
        _fibonacci_sphere(N, offset=1.1),
        _random_sphere(N, rng),
        _random_sphere(N, rng),
        _random_sphere(N, rng),
    ]
    
    # Allocate time per restart
    n_restarts = len(starts)
    time_per_restart = (deadline - time.time()) / n_restarts
    
    for idx, X0 in enumerate(starts):
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        
        # Scale iterations based on remaining time
        alloc_time = min(time_per_restart, remaining * 0.8)
        max_iter_lbfgs = max(100, int(alloc_time * 30))
        max_iter_polish = max(100, int(alloc_time * 50))
        
        X1, E1 = _optimize_lbfgs(X0, max_iter=max_iter_lbfgs, tol=1e-12, deadline=deadline)
        X2, E2 = _polish(X1, max_iter=max_iter_polish, tol=1e-15, deadline=deadline)
        
        if E2 < best_E and _validate(X2):
            best_E = E2
            best_X = X2.copy()
    
    if best_X is None:
        # Fallback to Fibonacci
        best_X = _fibonacci_sphere(N)
    
    # Final normalization
    norms = np.linalg.norm(best_X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_X = best_X / norms
    
    # Final validation
    if not _validate(best_X):
        best_X = _fibonacci_sphere(N)
        norms = np.linalg.norm(best_X, axis=1, keepdims=True)
        best_X = best_X / np.maximum(norms, 1e-15)
    
    return best_X
# EVOLVE-BLOCK-END
