import numpy as np
import time
from scipy.optimize import minimize

N = 150

def _energy_and_grad_flat(flat):
    """Energy and gradient for L-BFGS-B on flattened coordinates."""
    x = flat.reshape(N, 3)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    mask = dist_sq > 1e-12
    inv_dist_sq = np.zeros_like(dist_sq)
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    log_dist_sq = np.where(mask, np.log(dist_sq), 0.0)
    total_log_dist_sq = np.sum(log_dist_sq)
    energy = -0.25 * total_log_dist_sq
    return energy, grad.flatten()

def _project_to_sphere(x):
    norms = np.linalg.norm(x, axis=1)
    norms[norms < 1e-12] = 1e-12
    return x / norms[:, np.newaxis]

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _lbfgs_polish(x, time_budget):
    """Polish with L-BFGS-B using tangent space projection."""
    start = time.time()
    deadline = start + time_budget
    
    def obj_and_grad(flat):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        x = flat.reshape(N, 3)
        # Project to sphere
        x = _project_to_sphere(x)
        e, g = _energy_and_grad_flat(flat)
        # Compute gradient in tangent space
        # For sphere constraint, project gradient: g_tan = g - (g·x)x
        g_tan = g - np.sum(g * x, axis=1, keepdims=True) * x
        return e, g_tan.flatten()
    
    try:
        res = minimize(obj_and_grad, x.flatten(), jac=True, method='L-BFGS-B',
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
        if res.success and res.fun < float('inf'):
            x_new = _project_to_sphere(res.x.reshape(N, 3))
            return x_new
    except Exception:
        pass
    return x

def _momentum_optimize(x, start_time, time_limit):
    """Momentum-based optimization with adaptive step size."""
    momentum = 0.9
    v = np.zeros_like(x)
    step_size = 0.05
    min_step = 1e-8
    max_step = 0.2
    
    best_x = x.copy()
    best_energy = float('inf')
    
    def compute_energy_and_grad(points):
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        mask = dist_sq > 1e-12
        inv_dist_sq = np.zeros_like(dist_sq)
        inv_dist_sq[mask] = 1.0 / dist_sq[mask]
        grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        log_dist_sq = np.where(mask, np.log(dist_sq), 0.0)
        total_log_dist_sq = np.sum(log_dist_sq)
        energy = -0.25 * total_log_dist_sq
        return energy, grad
    
    current_energy, _ = compute_energy_and_grad(x)
    best_energy = current_energy
    best_x = x.copy()
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < time_limit:
        energy, grad = compute_energy_and_grad(x)
        
        v = momentum * v - step_size * grad
        x_new = x + v
        x_new = _project_to_sphere(x_new)
        
        new_energy, _ = compute_energy_and_grad(x_new)
        
        if new_energy < best_energy - 1e-12:
            x = x_new
            best_energy = new_energy
            best_x = x.copy()
            no_improve_count = 0
            step_size = min(max_step, step_size * 1.02)
        else:
            no_improve_count += 1
            step_size = max(min_step, step_size * 0.85)
            if no_improve_count >= max_no_improve:
                # Reset momentum and try smaller steps
                v = np.zeros_like(x)
                no_improve_count = 0
                if step_size <= min_step * 1.1:
                    break
    
    return best_x, best_energy

def construct():
    start_time = time.time()
    total_time_limit = 145.0  # Safety margin for 150s limit
    
    # Phase 1: Fibonacci spiral initialization
    x = _fibonacci_spiral(N)
    
    # Phase 2: Momentum optimization
    phase1_limit = start_time + 60.0
    x, best_energy = _momentum_optimize(x, start_time, phase1_limit)
    
    # Phase 3: L-BFGS polish
    remaining = total_time_limit - (time.time() - start_time)
    if remaining > 10:
        x = _lbfgs_polish(x, remaining - 5)
    
    # Final validation
    x = _project_to_sphere(x)
    return x

# EVOLVE-BLOCK-END
