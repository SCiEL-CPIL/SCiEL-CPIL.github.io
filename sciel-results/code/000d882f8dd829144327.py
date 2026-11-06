import numpy as np
from scipy.optimize import minimize
import time
import math

N = 150

def _energy_and_grad(x_flat):
    """
    Calculate energy and gradient (projected to tangent space).
    """
    x = x_flat.reshape(N, 3)
    # Project to sphere
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    x_proj = x / norms
    
    # Pairwise squared distances
    diff = x_proj[:, np.newaxis, :] - x_proj[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1) + 1e-30
    dist = np.sqrt(dist_sq)
    
    # Mask for i < j
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d_pairs = dist[mask]
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum log(d_ij)
    e = -math.fsum(np.log(d_pairs).tolist())
    
    # Gradient: dE/dx_i = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    weights = 1.0 / dist_sq
    np.fill_diagonal(weights, 0.0)
    
    w_sum = np.sum(weights, axis=1)
    w_x = weights @ x_proj
    
    # Raw gradient
    grad_raw = -x_proj * w_sum[:, np.newaxis] + w_x
    
    # Project to tangent space: g_t = g - (g . x) x
    g_dot_x = np.sum(grad_raw * x_proj, axis=1, keepdims=True)
    grad_tangent = grad_raw - g_dot_x * x_proj
    
    return e, grad_tangent.flatten()

def _polish(x, max_iter=200, tol=1e-14):
    """
    Gradient descent with backtracking line search on the sphere.
    """
    x = x.copy()
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    def energy(x):
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1) + 1e-30
        d_pairs = np.sqrt(dist_sq[mask])
        return -math.fsum(np.log(d_pairs).tolist())
    
    def grad(x):
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1) + 1e-30
        weights = 1.0 / dist_sq
        np.fill_diagonal(weights, 0.0)
        w_sum = np.sum(weights, axis=1)
        w_x = weights @ x
        g = -x * w_sum[:, np.newaxis] + w_x
        g_dot_x = np.sum(g * x, axis=1, keepdims=True)
        g_tangent = g - g_dot_x * x
        return g_tangent

    curr_e = energy(x)
    step = 0.05
    for it in range(max_iter):
        g = grad(x)
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break
        
        # Adaptive step size: start smaller for stability
        alpha = step
        improved = False
        for _ in range(30):
            x_new = x - alpha * g
            norms_new = np.linalg.norm(x_new, axis=1, keepdims=True)
            norms_new = np.where(norms_new < 1e-12, 1.0, norms_new)
            x_new = x_new / norms_new
            
            new_e = energy(x_new)
            if new_e < curr_e - 1e-4 * alpha * g_norm**2:
                x = x_new
                curr_e = new_e
                improved = True
                break
            alpha *= 0.5
        if not improved:
            break
        # Gradually decrease step
        if it % 20 == 19 and step > 0.01:
            step *= 0.8
            
    return x

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    def time_left():
        return deadline - time.time()

    # Initializations
    rng = np.random.default_rng(42)
    all_inits = []
    
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    all_inits.append(x_fib)
    
    # 2. Fibonacci + small noise (multiple scales)
    for noise_scale in [0.005, 0.02, 0.05]:
        noise = rng.normal(0, noise_scale, (N, 3))
        x_fib_noise = x_fib + noise
        x_fib_noise /= np.linalg.norm(x_fib_noise, axis=1, keepdims=True)
        all_inits.append(x_fib_noise)
    
    # 3. Random initializations
    num_randoms = 25
    for k in range(num_randoms):
        x_rand = rng.normal(size=(N, 3))
        x_rand /= np.linalg.norm(x_rand, axis=1, keepdims=True)
        all_inits.append(x_rand)

    best_x = all_inits[0].copy()
    best_energy = _energy_and_grad(best_x.flatten())[0]

    for idx, x0 in enumerate(all_inits):
        if time_left() < 10.0:
            break
        
        try:
            # L-BFGS-B with analytic gradient
            result = minimize(
                _energy_and_grad, 
                x0.flatten(), 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14, 'maxls': 50},
            )
            
            x_opt = result.x.reshape(N, 3)
            x_opt /= np.linalg.norm(x_opt, axis=1, keepdims=True)
            
            # Polish with gradient descent
            if time_left() > 5.0:
                x_opt = _polish(x_opt, max_iter=200, tol=1e-15)
            
            e_opt = _energy_and_grad(x_opt.flatten())[0]
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_x = x_opt.copy()
                
        except Exception:
            continue

    # Final verification: ensure all points are on unit sphere and finite
    norms = np.linalg.norm(best_x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    best_x = best_x / norms
    best_x = np.where(np.isfinite(best_x), best_x, 0.0)
    # Re-normalize after any zeroing
    norms = np.linalg.norm(best_x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    best_x = best_x / norms
    
    return best_x
