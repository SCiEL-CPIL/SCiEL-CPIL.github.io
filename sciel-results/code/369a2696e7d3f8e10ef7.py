import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def _halton_sphere(n):
    """Generate n points on the unit sphere using a 3D Halton sequence."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    
    points = []
    for i in range(1, n + 1):
        x = halton(i, 2)
        y = halton(i, 3)
        z = halton(i, 5)
        # Map from [0,1]^3 to sphere
        # Use a simple method: reject sampling or direct mapping
        # Direct mapping: use spherical coordinates
        theta = 2 * np.pi * x
        phi = np.arccos(2 * y - 1)
        r = 1.0
        px = r * np.sin(phi) * np.cos(theta)
        py = r * np.sin(phi) * np.sin(theta)
        pz = r * np.cos(phi)
        points.append([px, py, pz])
    
    return np.array(points)

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _energy_and_grad(X):
    """
    Compute logarithmic energy and its gradient.
    X: (N, 3) array of points on the unit sphere.
    Returns energy (scalar) and grad (N, 3) array.
    """
    # Pairwise differences
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    # Squared distances
    d2 = np.sum(diff**2, axis=2)
    
    # Avoid division by zero for self-distances
    np.fill_diagonal(d2, 1.0)
    
    # Weights 1/d2
    w = 1.0 / d2
    np.fill_diagonal(w, 0.0)
    
    # Energy: sum_{i<j} log(1/d_ij)
    # log(1/d) = -0.5 * log(d2)
    # Sum_{i<j} = 0.5 * Sum_{i!=j}
    # E = -0.25 * sum_{i!=j} log(d2_ij)
    
    log_d2 = np.log(d2)
    np.fill_diagonal(log_d2, 0.0)
    
    energy = -0.25 * np.sum(log_d2)
    
    # Gradient of energy with respect to x_i
    # dE/dx_i = - sum_{j != i} (x_i - x_j) / d2_ij
    grad = -np.sum(diff * w[:, :, np.newaxis], axis=1)
    
    return energy, grad

def _project_tangent(X, G):
    """Project gradient G to the tangent space of the sphere at each point X."""
    # Tangent space: g_tan = g - (g . x) x
    dot = np.sum(G * X, axis=1)
    G_tan = G - dot[:, np.newaxis] * X
    return G_tan

def _exp_map(X, V):
    """
    Exponential map on the unit sphere.
    X: current points (N, 3)
    V: tangent vectors (N, 3)
    Returns new points on the sphere.
    """
    theta = np.linalg.norm(V, axis=1)
    # Avoid division by zero
    theta_safe = np.where(theta > 1e-12, theta, 1.0)
    V_norm = V / theta_safe[:, np.newaxis]
    
    # exp_map: x * cos(theta) + v/|v| * sin(theta)
    X_new = X * np.cos(theta_safe)[:, np.newaxis] + V_norm * np.sin(theta_safe)[:, np.newaxis]
    
    # Re-normalize for numerical stability
    X_new = X_new / np.linalg.norm(X_new, axis=1)[:, np.newaxis]
    
    return X_new

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin for safety
    
    # Initial guess: Halton sequence for better dispersion
    x0 = _halton_sphere(N)
    
    # Also try Fibonacci and pick the better one
    x0_fib = _fibonacci_sphere(N)
    e_halton, _ = _energy_and_grad(x0)
    e_fib, _ = _energy_and_grad(x0_fib)
    
    if e_fib < e_halton:
        x0 = x0_fib
        e0 = e_fib
    else:
        e0 = e_halton
    
    best_x = x0.copy()
    best_e = e0
    
    # Phase 1: L-BFGS-B with penalty for norm constraint
    penalty_coeff = 100.0
    
    def obj_penalized(x_flat):
        X = x_flat.reshape(N, 3)
        e, g = _energy_and_grad(X)
        norms2 = np.sum(X**2, axis=1)
        pen = 0.5 * penalty_coeff * np.sum((norms2 - 1.0)**2)
        return e + pen
    
    def grad_penalized(x_flat):
        X = x_flat.reshape(N, 3)
        e, g = _energy_and_grad(X)
        norms2 = np.sum(X**2, axis=1)
        g_pen = g + penalty_coeff * (norms2 - 1.0)[:, np.newaxis] * X
        return g_pen
    
    try:
        res = minimize(
            obj_penalized,
            x0.flatten(),
            jac=grad_penalized,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        x_res = res.x.reshape(N, 3)
        x_res = x_res / np.linalg.norm(x_res, axis=1)[:, np.newaxis]
        e_res, _ = _energy_and_grad(x_res)
        if e_res < best_e:
            best_e = e_res
            best_x = x_res.copy()
    except Exception:
        pass
    
    # Phase 2: Riemannian Conjugate Gradient on the sphere
    if time.time() < deadline - 20.0:
        X_curr = best_x.copy()
        e_curr, g_curr = _energy_and_grad(X_curr)
        
        # Project gradient to tangent space
        g_tan = _project_tangent(X_curr, g_curr)
        
        # Initialize momentum
        p_curr = -g_tan.copy()
        
        # Conjugate gradient parameters
        beta = 0.0
        max_cg_iter = 200
        
        for cg_iter in range(max_cg_iter):
            if time.time() > deadline:
                break
            
            # Line search for step size
            alpha = 1.0
            improved = False
            for _ in range(50):
                X_new = _exp_map(X_curr, alpha * p_curr)
                e_new, g_new = _energy_and_grad(X_new)
                
                if e_new < e_curr:
                    # Check Armijo condition
                    if e_new <= e_curr + 1e-4 * alpha * np.sum(g_tan * p_curr):
                        improved = True
                        break
                alpha *= 0.5
            
            if not improved:
                # Reset momentum
                p_curr = -g_tan.copy()
                beta = 0.0
                continue
            
            # Update
            X_prev = X_curr
            e_prev = e_curr
            g_prev = g_tan
            
            X_curr = X_new
            e_curr = e_new
            g_curr, _ = _energy_and_grad(X_curr)
            g_tan = _project_tangent(X_curr, g_curr)
            
            # Check convergence
            if np.linalg.norm(g_tan) < 1e-10:
                break
            
            # Update beta (Polak-Ribiere)
            beta_new = np.sum(g_tan * (g_tan - g_prev)) / (np.sum(g_prev * g_prev) + 1e-12)
            beta = max(0.0, beta_new)
            
            # Update search direction
            p_curr = -g_tan + beta * p_curr
            
            # Update best
            if e_curr < best_e:
                best_e = e_curr
                best_x = X_curr.copy()
    
    # Phase 3: Fine-tuning with small L-BFGS-B steps
    if time.time() < deadline - 5.0:
        try:
            res2 = minimize(
                obj_penalized,
                best_x.flatten(),
                jac=grad_penalized,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            x_res2 = res2.x.reshape(N, 3)
            x_res2 = x_res2 / np.linalg.norm(x_res2, axis=1)[:, np.newaxis]
            e_res2, _ = _energy_and_grad(x_res2)
            if e_res2 < best_e:
                best_e = e_res2
                best_x = x_res2.copy()
        except Exception:
            pass
    
    # Final normalization
    best_x = best_x / np.linalg.norm(best_x, axis=1)[:, np.newaxis]
    
    return best_x
