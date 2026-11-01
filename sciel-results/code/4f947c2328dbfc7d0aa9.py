"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _energy_and_grad(X):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    and its gradient w.r.t. X (shape (N,3)).
    """
    dots = X @ X.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    
    iu = np.triu_indices(N, 1)
    log_d2 = np.log(d2[iu])
    E = -0.5 * math.fsum(log_d2.tolist())
    
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    row_sums = inv_d2.sum(axis=1)
    term2 = inv_d2 @ X
    grad = X * row_sums[:, None] - term2
    
    return E, grad

def _retract(X, step):
    """Riemannian retraction: move along tangent direction, then project back to sphere."""
    step_dot_X = np.sum(step * X, axis=1, keepdims=True)
    tangent_step = step - step_dot_X * X
    X_new = X + tangent_step
    norms = np.linalg.norm(X_new, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    X_new = X_new / norms
    return X_new

def _optimize_lbfgs(X0, max_iter=5000, tol=1e-12, deadline=None):
    """L-BFGS optimization on the sphere using retraction with adaptive memory."""
    X = X0.copy()
    n = X.shape[0]
    m = 30
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
                rho_i = 1.0 / max(np.dot(s_i, y_i), 1e-15)
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
        
        # Backtracking line search with stronger Wolfe conditions
        step_size = 1.0
        g_flat_dir = np.dot(g_flat, direction.ravel())
        
        if g_flat_dir >= 0:
            direction = -g
            g_flat_dir = np.dot(g_flat, direction.ravel())
        
        X_trial = None
        E_trial = None
        g_trial = None
        
        for ls in range(50):
            X_cand = _retract(X, step_size * direction)
            E_cand, g_cand = _energy_and_grad(X_cand)
            
            if E_cand <= E + 1e-4 * step_size * g_flat_dir:
                g_cand_flat = g_cand.ravel()
                if np.dot(g_cand_flat, direction.ravel()) <= 0.9 * g_flat_dir:
                    X_trial = X_cand
                    E_trial = E_cand
                    g_trial = g_cand
                    break
            step_size *= 0.5
        
        if X_trial is None:
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

def _polish(X0, max_iter=50000, tol=1e-14, deadline=None):
    """High-precision steepest descent polish with adaptive step size."""
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
        
        direction = -g / g_norm
        
        improved = False
        for ls in range(50):
            X_trial = _retract(X, step_size * direction)
            E_trial, g_trial = _energy_and_grad(X_trial)
            
            if E_trial < E:
                X = X_trial
                E = E_trial
                g = g_trial
                improved = True
                if E < best_E:
                    best_E = E
                    best_X = X.copy()
                break
            step_size *= 0.5
        
        if not improved:
            step_size *= 0.1
            if step_size < 1e-20:
                break
        else:
            step_size = min(step_size * 1.5, 1e-1)
    
    return best_X, best_E

def _newton_polish(X0, max_iter=100, deadline=None):
    """Newton-type polish using Hessian approximation for faster convergence."""
    X = X0.copy()
    E, g = _energy_and_grad(X)
    best_E = E
    best_X = X.copy()
    
    # Approximate Hessian via finite differences of gradient
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        g_flat = g.ravel()
        g_norm = np.linalg.norm(g_flat)
        if g_norm < 1e-14:
            break
        
        # Simple Newton step: use diagonal Hessian approximation
        # H_ii ≈ (g_i(x+ε) - g_i(x-ε)) / (2ε) for each component
        eps = 1e-7
        H_diag = np.zeros_like(g_flat)
        
        for i in range(0, len(g_flat), 100):  # Process in batches for speed
            if deadline is not None and time.time() > deadline:
                break
            end = min(i + 100, len(g_flat))
            for j in range(i, end):
                X_plus = X.copy()
                X_plus.flat[j] += eps
                X_plus = X_plus / np.linalg.norm(X_plus, axis=1, keepdims=True)
                _, g_plus = _energy_and_grad(X_plus)
                
                X_minus = X.copy()
                X_minus.flat[j] -= eps
                X_minus = X_minus / np.linalg.norm(X_minus, axis=1, keepdims=True)
                _, g_minus = _energy_and_grad(X_minus)
                
                H_diag[j] = (g_plus.flat[j] - g_minus.flat[j]) / (2 * eps)
        
        # Newton step with damping
        H_diag_safe = np.where(np.abs(H_diag) > 1e-10, H_diag, 1.0)
        newton_step = -g_flat / H_diag_safe
        newton_step = newton_step.reshape(N, 3)
        
        # Line search on Newton step
        step_size = 1.0
        for ls in range(20):
            X_trial = _retract(X, step_size * newton_step)
            E_trial, g_trial = _energy_and_grad(X_trial)
            
            if E_trial < E:
                X = X_trial
                E = E_trial
                g = g_trial
                if E < best_E:
                    best_E = E
                    best_X = X.copy()
                break
            step_size *= 0.5
        
        if step_size < 1e-10:
            break
    
    return best_X, best_E

def _fibonacci_sphere(n):
    """Generate n points on the sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed):
    """Generate n random points on the sphere."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 3)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norms

def _icosahedron_seed(n):
    """Generate points based on icosahedron vertices with jitter."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    rng = np.random.RandomState(42)
    X = np.zeros((n, 3))
    for i in range(n):
        v = verts[rng.randint(12)]
        noise = rng.randn(3) * 0.3
        X[i] = v + noise
        X[i] = X[i] / np.linalg.norm(X[i])
    return X

def _golden_ratio_seed(n):
    """Generate points using golden ratio distribution."""
    rng = np.random.RandomState(99999)
    X = np.zeros((n, 3))
    for i in range(n):
        z = 1.0 - 2.0 * i / n
        phi = 2.0 * np.pi * i * 0.618033988749895
        r = np.sqrt(1.0 - z * z)
        X[i] = [r * np.cos(phi), r * np.sin(phi), z]
    # Add small random perturbation
    noise = rng.randn(n, 3) * 0.01
    X = X + noise
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X

def construct():
    """
    Construct 100 points on the unit sphere to minimize logarithmic energy.
    """
    deadline = time.time() + 140.0
    
    best_X = None
    best_E = float('inf')
    
    # Multiple starting points
    starts = [
        _fibonacci_sphere(N),
        _random_sphere(N, 12345),
        _random_sphere(N, 67890),
        _random_sphere(N, 11111),
        _random_sphere(N, 54321),
        _icosahedron_seed(N),
        _golden_ratio_seed(N),
    ]
    
    for start in starts:
        if time.time() > deadline - 30.0:
            break
        
        # L-BFGS phase
        X, E = _optimize_lbfgs(start, max_iter=3000, tol=1e-12, deadline=deadline)
        
        if E < best_E:
            best_E = E
            best_X = X.copy()
        
        # Polish phase
        if time.time() > deadline - 15.0:
            break
        X, E = _polish(X, max_iter=30000, tol=1e-14, deadline=deadline)
        
        if E < best_E:
            best_E = E
            best_X = X.copy()
        
        # Newton polish phase
        if time.time() > deadline - 5.0:
            break
        X, E = _newton_polish(X, max_iter=50, deadline=deadline)
        
        if E < best_E:
            best_E = E
            best_X = X.copy()
    
    if best_X is None:
        best_X = _fibonacci_sphere(N)
    
    # Final normalization
    norms = np.linalg.norm(best_X, axis=1)
    best_X = best_X / norms[:, None]
    
    # Check for very close points
    min_dist = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_X[i] - best_X[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 1e-9:
        return _fibonacci_sphere(N)
    
    return best_X
# EVOLVE-BLOCK-END
