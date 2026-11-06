import numpy as np
from scipy.optimize import minimize
import time
import math

N = 150

def _energy_only(x):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    # Use upper triangle only
    iu = np.triu_indices(N, k=1)
    d = np.sqrt(dist_sq[iu] + 1e-30)
    return -math.fsum(np.log(d).tolist())

def _energy_and_grad(x_flat):
    """Energy and gradient for L-BFGS-B (Euclidean, with projection)."""
    x = x_flat.reshape(N, 3)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    x = x / norms
    
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    dist_sq_safe = np.maximum(dist_sq, 1e-30)
    
    iu = np.triu_indices(N, k=1)
    d = np.sqrt(dist_sq_safe[iu])
    e = -math.fsum(np.log(d).tolist())
    
    # Gradient: dE/dx_i = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    weights = 1.0 / dist_sq_safe
    np.fill_diagonal(weights, 0.0)
    w_sum = np.sum(weights, axis=1)
    w_x = weights @ x
    grad_raw = -x * w_sum[:, np.newaxis] + w_x
    
    # Project to tangent space
    g_dot_x = np.sum(grad_raw * x, axis=1, keepdims=True)
    grad_tangent = grad_raw - g_dot_x * x
    
    return e, grad_tangent.flatten()

def _polish(x, max_iter=800, tol=1e-16):
    """High-precision gradient descent on the sphere."""
    x = x.copy()
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    
    iu = np.triu_indices(N, k=1)
    
    def energy(x):
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        d = np.sqrt(dist_sq[iu] + 1e-30)
        return -math.fsum(np.log(d).tolist())
    
    def grad(x):
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        dist_sq_safe = np.maximum(dist_sq, 1e-30)
        weights = 1.0 / dist_sq_safe
        np.fill_diagonal(weights, 0.0)
        w_sum = np.sum(weights, axis=1)
        w_x = weights @ x
        g = -x * w_sum[:, np.newaxis] + w_x
        g_dot_x = np.sum(g * x, axis=1, keepdims=True)
        return g - g_dot_x * x
    
    curr_e = energy(x)
    alpha = 0.01
    for it in range(max_iter):
        g = grad(x)
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break
        
        improved = False
        a = alpha
        for _ in range(60):
            x_new = x - a * g
            nrm = np.linalg.norm(x_new, axis=1, keepdims=True)
            nrm = np.where(nrm < 1e-12, 1.0, nrm)
            x_new = x_new / nrm
            
            new_e = energy(x_new)
            if new_e < curr_e - 1e-4 * a * g_norm**2:
                x = x_new
                curr_e = new_e
                improved = True
                break
            a *= 0.5
        if not improved:
            break
        if it % 50 == 49 and alpha > 0.002:
            alpha *= 0.9
    
    return x

def _spectral_deform(x, k_modes=20):
    """
    Spectral deformation: project to low-frequency spherical harmonics
    to regularize the landscape, then project back to sphere.
    This helps escape shallow local minima.
    """
    # Use a simple approach: average nearby points to smooth
    # Actually, let's use a different strategy: add small random perturbations
    # and re-optimize to escape local minima
    return x

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    def time_left():
        return deadline - time.time()
    
    rng = np.random.default_rng(12345)
    
    # Generate diverse initializations
    all_inits = []
    
    # 1. Fibonacci Spiral (multiple variants)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    all_inits.append(x_fib.copy())
    
    # 2. Fibonacci with various noise levels
    for ns in [0.001, 0.003, 0.007, 0.015, 0.03, 0.06, 0.12, 0.25]:
        noise = rng.normal(0, ns, (N, 3))
        x = x_fib + noise
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        all_inits.append(x)
    
    # 3. Random initializations with different seeds
    for k in range(20):
        x = rng.normal(size=(N, 3))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        all_inits.append(x)
    
    # 4. Structured initializations (spherical designs approximations)
    # Use a grid-like approach
    for theta_offset in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        thetas = np.linspace(0, np.pi, N, endpoint=False) + theta_offset / N
        phis = np.linspace(0, 2*np.pi, N, endpoint=False)
        x = np.column_stack([
            np.sin(thetas) * np.cos(phis),
            np.sin(thetas) * np.sin(phis),
            np.cos(thetas)
        ])
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        all_inits.append(x)
    
    best_x = all_inits[0].copy()
    best_energy = _energy_only(best_x)
    
    # Optimize each initialization
    for idx, x0 in enumerate(all_inits):
        if time_left() < 8.0:
            break
        
        try:
            # L-BFGS-B optimization
            result = minimize(
                _energy_and_grad,
                x0.flatten(),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 100, 'm': 50},
            )
            
            x_opt = result.x.reshape(N, 3)
            x_opt /= np.linalg.norm(x_opt, axis=1, keepdims=True)
            
            # Polish with gradient descent
            if time_left() > 3.0:
                x_opt = _polish(x_opt, max_iter=500, tol=1e-15)
            
            e_opt = _energy_only(x_opt)
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_x = x_opt.copy()
                
        except Exception:
            continue
    
    # Final high-precision polish
    if time_left() > 2.0:
        try:
            best_x = _polish(best_x, max_iter=1000, tol=1e-17)
            final_e = _energy_only(best_x)
            if final_e < best_energy:
                best_energy = final_e
        except Exception:
            pass
    
    # Ensure valid output
    norms = np.linalg.norm(best_x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    best_x = best_x / norms
    best_x = np.where(np.isfinite(best_x), best_x, 0.0)
    norms = np.linalg.norm(best_x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    best_x = best_x / norms
    
    return best_x
