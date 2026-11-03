import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _remaining(deadline):
    return max(0.0, deadline - time.time())

def _energy_and_grad(pts):
    """
    Compute the logarithmic energy E = sum_{i<j} log(1/||x_i - x_j||)
    and its Euclidean gradient.
    
    E = -0.5 * sum_{i<j} log(||x_i - x_j||^2)
    grad_E_i = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    """
    n = pts.shape[0]
    # Squared distances
    # D2[i, j] = ||x_i - x_j||^2
    D2 = np.sum(pts**2, axis=1)[:, None] + np.sum(pts**2, axis=1)[None, :] - 2.0 * (pts @ pts.T)
    
    # Clip to avoid log(0) or division by zero, though points should be distinct
    D2 = np.maximum(D2, 1e-15)
    np.fill_diagonal(D2, 1.0) # Diagonal doesn't contribute to energy or gradient sum directly in the loop, but safe for matrix ops
    
    # Energy
    # Sum over i < j of log(1/sqrt(D2_ij)) = -0.5 * sum_{i<j} log(D2_ij)
    # Sum_{i<j} log(D2_ij) = 0.5 * (Sum_{i != j} log(D2_ij))
    # So E = -0.25 * Sum_{i != j} log(D2_ij)
    log_D2 = np.log(D2)
    np.fill_diagonal(log_D2, 0.0)
    energy = -0.25 * np.sum(log_D2)
    
    # Gradient
    # grad_i = sum_{j != i} (x_i - x_j) / D2_ij
    # Let Diff[i, j, k] = pts[i, k] - pts[j, k]
    Diff = pts[:, None, :] - pts[None, :, :]
    
    # We need to divide each pair difference by D2_ij
    # D2 shape (N, N). We want to multiply Diff[i, j, :] by 1/D2[i, j]
    InvD2 = 1.0 / D2
    np.fill_diagonal(InvD2, 0.0) # No self-interaction
    
    # Gradient component for point i is sum_j (x_i - x_j) / D2_ij
    # This is a matrix multiplication of Diff (N, N, 3) by InvD2 (N, N) along the j-axis
    grad = np.sum(Diff * InvD2[:, :, None], axis=1)
    
    return energy, grad

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _fibonacci_init():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(X)

def _random_init(seed=None):
    if seed is not None:
        np.random.seed(seed)
    pts = np.random.randn(N, 3)
    return _project_to_sphere(pts)

def _sobol_init():
    # Simple quasi-random initialization using Halton or similar if scipy not available for Sobol directly in older versions, 
    # but scipy.stats.qmc.LatinHypercube or Sobol is available in recent SciPy.
    # Let's use a deterministic low-discrepancy sequence if possible, or just random with fixed seeds.
    # Since we need robustness, let's stick to a few deterministic random seeds and Fibonacci.
    # Actually, let's try a simple grid-based or spiral-based variation.
    # For now, random with different seeds is sufficient.
    return _random_init(seed=42)

def _optimize_from_start(X0, deadline, method='L-BFGS-B', tol=1e-12):
    """
    Optimize points on the sphere using L-BFGS-B on the embedding space,
    with a projection step or penalty. 
    However, L-BFGS-B doesn't handle constraints well unless we parameterize.
    
    Better approach: Use the gradient on the sphere.
    We can use scipy.optimize.minimize with a custom method or just implement a simple gradient descent with line search.
    Or, use the fact that we can minimize on the manifold.
    
    Let's use a simple Riemannian Gradient Descent with Barzilai-Borwein step size or similar,
    or just use scipy.optimize.minimize with 'BFGS' on the flat space and project? No, that's not standard.
    
    Standard approach: 
    Minimize f(x) subject to ||x_i|| = 1.
    Lagrangian: L = f(x) + sum lambda_i (||x_i||^2 - 1).
    Gradient: grad_f + 2 lambda_i x_i = 0 => lambda_i = - (grad_f . x_i) / 2.
    So the Riemannian gradient is grad_f - (grad_f . x_i) x_i.
    
    We can use scipy.optimize.minimize with a custom function that projects? No.
    Let's implement a simple gradient descent with adaptive step size on the sphere.
    """
    
    def obj_and_grad(x_flat):
        pts = x_flat.reshape(N, 3)
        pts = _project_to_sphere(pts)
        e, g = _energy_and_grad(pts)
        # Project gradient to tangent space
        dot_g_x = np.sum(g * pts, axis=1, keepdims=True)
        g_tan = g - dot_g_x * pts
        return e, g_tan.flatten()
    
    # We will use L-BFGS-B on the flattened vector, but we need to ensure the function respects the manifold.
    # A common trick is to minimize on R^{3N} and project in the objective, but the gradient must be consistent.
    # If we project x to x' in the objective, the gradient of the projected objective is not simply the projected gradient.
    
    # Better: Use a dedicated manifold optimizer or simple GD.
    # Let's use simple GD with momentum and adaptive learning rate.
    
    X = X0.copy()
    V = np.zeros_like(X)
    lr = 0.01
    momentum = 0.9
    best_e, best_X = _energy_and_grad(X)[0], X.copy()
    
    # Calculate initial gradient
    e, g = _energy_and_grad(X)
    dot_g_x = np.sum(g * X, axis=1, keepdims=True)
    g_tan = g - dot_g_x * X
    
    # Adaptive LR: Barzilai-Borwein style or simple decay
    for i in range(20000):
        if time.time() > deadline:
            break
        
        # Update velocity
        V = momentum * V + lr * g_tan
        X_new = X + V
        X_new = _project_to_sphere(X_new)
        
        # Compute new energy
        e_new, g_new = _energy_and_grad(X_new)
        
        # Simple line search or just accept if improved
        if e_new < best_e:
            best_e = e_new
            best_X = X_new.copy()
        
        # Accept step if energy didn't increase too much (or always accept for simplicity with small lr)
        # Let's use a simple backtracking line search
        alpha = 1.0
        while e_new > e + 1e-4 * alpha * np.sum(g_tan * V):
            alpha *= 0.5
            if alpha < 1e-6:
                break
            V_trial = momentum * V + lr * g_tan
            X_trial = X + alpha * V_trial
            X_trial = _project_to_sphere(X_trial)
            e_trial, _ = _energy_and_grad(X_trial)
            if e_trial < e:
                e_new = e_trial
                X_new = X_trial
                break
        
        # Update
        X = X_new
        e = e_new
        
        # Update gradient
        e, g = _energy_and_grad(X)
        dot_g_x = np.sum(g * X, axis=1, keepdims=True)
        g_tan = g - dot_g_x * X
        
        # Decay learning rate
        lr *= 0.999
        
    return best_X, best_e

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # 1. Initial guesses
    X_fib = _fibonacci_init()
    X_rand1 = _random_init(seed=1)
    X_rand2 = _random_init(seed=123)
    
    best_X = X_fib
    best_E = _energy_and_grad(X_fib)[0]
    
    candidates = [X_fib, X_rand1, X_rand2]
    
    # 2. Optimize each candidate
    for idx, X0 in enumerate(candidates):
        if time.time() > deadline - 10.0:
            break
        
        # Use a tighter deadline for each optimization to allow multiple
        sub_deadline = min(deadline - 5.0, start_time + 35.0 * (idx + 1)) # Roughly 35s each? No, total 110s.
        # Let's give each about 30s if we have 3 candidates.
        # Actually, just check remaining time.
        
        X_opt, E_opt = _optimize_from_start(X0, deadline - 2.0)
        
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt
    
    # 3. Final polish with a more precise method if time permits
    # Use scipy.optimize.minimize with L-BFGS-B on the flat space?
    # Or just run more iterations of the GD with very small lr.
    
    if time.time() < deadline - 5.0:
        # Polish
        X_polish = best_X.copy()
        V = np.zeros_like(X_polish)
        lr = 1e-4
        momentum = 0.9
        
        e, g = _energy_and_grad(X_polish)
        dot_g_x = np.sum(g * X_polish, axis=1, keepdims=True)
        g_tan = g - dot_g_x * X_polish
        
        for i in range(5000):
            if time.time() > deadline - 1.0:
                break
                
            V = momentum * V + lr * g_tan
            X_new = X_polish + V
            X_new = _project_to_sphere(X_new)
            
            e_new, g_new = _energy_and_grad(X_new)
            
            if e_new < best_E:
                best_E = e_new
                best_X = X_new.copy()
            
            # Line search
            alpha = 1.0
            while e_new > e + 1e-4 * alpha * np.sum(g_tan * V):
                alpha *= 0.5
                if alpha < 1e-6:
                    break
                V_trial = momentum * V + lr * g_tan
                X_trial = X_polish + alpha * V_trial
                X_trial = _project_to_sphere(X_trial)
                e_trial, _ = _energy_and_grad(X_trial)
                if e_trial < e:
                    e_new = e_trial
                    X_new = X_trial
                    break
            
            X_polish = X_new
            e = e_new
            
            e, g = _energy_and_grad(X_polish)
            dot_g_x = np.sum(g * X_polish, axis=1, keepdims=True)
            g_tan = g - dot_g_x * X_polish
            
            lr *= 0.99
    
    # Final check for validity
    norms = np.linalg.norm(best_X, axis=1)
    if np.any(norms < 1e-9) or np.any(~np.isfinite(best_X)):
        return X_fib # Fallback
    
    return best_X
