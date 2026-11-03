import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150
TIME_LIMIT = 115.0  # Safety margin before 120s kill

def _energy_and_grad(pos):
    """
    Compute logarithmic energy and gradient for points on unit sphere.
    
    pos: (N, 3) array of points on unit sphere.
    Returns (energy, grad) where grad is the unconstrained gradient w.r.t. pos.
    Energy is sum_{i<j} log(1 / |x_i - x_j|).
    """
    # Pairwise differences
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero for self-pairs
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    # Clip to avoid log(0) if points are extremely close (should not happen with proper init/step)
    d = np.clip(d, 1e-12, None)
    energy = -np.sum(np.log(d))
    
    # Gradient computation:
    # d/dp_i [ -log(|p_i - p_j|) ] = -(p_i - p_j) / |p_i - p_j|^2
    # grad_i = sum_{j != i} (p_j - p_i) / |p_i - p_j|^2
    #         = sum_{j != i} p_j / d_ij^2 - p_i * sum_{j != i} 1/d_ij^2
    
    inv_d2 = 1.0 / dist_sq
    np.fill_diagonal(inv_d2, 0.0) # Exclude self-interaction for gradient
    
    # sum_j p_j / d_ij^2 for each i
    sum_pj_over_d2 = inv_d2 @ pos
    # sum_j 1/d_ij^2 for each i
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    
    grad = sum_pj_over_d2 - pos * sum_inv_d2[:, np.newaxis]
    
    return energy, grad

def _project_to_sphere(pos):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pos, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pos / norms

def _penalty_objective(pos, lambda_pen=1e5):
    """
    Objective function with penalty for sphere constraint.
    pos: (N, 3) array.
    """
    pos_flat = pos.reshape(-1)
    pos_reshaped = pos_flat.reshape(N, 3)
    
    energy, grad_energy = _energy_and_grad(pos_reshaped)
    
    # Penalty term: sum (||x_i||^2 - 1)^2
    norms_sq = np.sum(pos_reshaped * pos_reshaped, axis=1)
    penalty = np.sum((norms_sq - 1.0) ** 2)
    
    # Gradient of penalty: d/dx_i (||x_i||^2 - 1)^2 = 2(||x_i||^2 - 1) * 2 x_i = 4 x_i (||x_i||^2 - 1)
    grad_penalty = 4.0 * pos_reshaped * (norms_sq - 1.0)[:, np.newaxis]
    
    total_energy = energy + lambda_pen * penalty
    total_grad = grad_energy + lambda_pen * grad_penalty
    
    return total_energy, total_grad.flatten()

def _optimize_lbfgs(pos_init, time_budget):
    """
    Optimize a single starting configuration using L-BFGS-B with penalty method.
    """
    start_time = time.time()
    pos = _project_to_sphere(pos_init.copy())
    
    # Initial energy check
    current_energy, _ = _energy_and_grad(pos)
    best_pos = pos.copy()
    best_energy = current_energy
    
    # We use L-BFGS-B on the flat vector
    x0 = pos.flatten()
    
    # Callback to check time and update best
    def callback(xk):
        nonlocal best_pos, best_energy
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            return True # Stop optimization
        
        pos_k = _project_to_sphere(xk.reshape(N, 3))
        e_k, _ = _energy_and_grad(pos_k)
        if e_k < best_energy:
            best_energy = e_k
            best_pos = pos_k.copy()
        return False

    try:
        # Use L-BFGS-B
        # maxiter is high, but we rely on time limit via callback
        res = minimize(
            _penalty_objective, 
            x0, 
            jac=True, 
            method='L-BFGS-B', 
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 10000}
        )
        
        # Final check after optimization
        if res.x is not None:
            pos_final = _project_to_sphere(res.x.reshape(N, 3))
            e_final, _ = _energy_and_grad(pos_final)
            if e_final < best_energy:
                best_energy = e_final
                best_pos = pos_final.copy()
                
    except Exception as e:
        # If L-BFGS fails, fall back to current best
        pass
        
    return _project_to_sphere(best_pos), best_energy

def _optimize(pos_init, time_budget):
    """
    Wrapper for optimization. We use L-BFGS-B now.
    """
    return _optimize_lbfgs(pos_init, time_budget)

def construct():
    start_time = time.time()
    
    # Generate a list of starting configurations
    starts = []
    
    # 1. Fibonacci spiral (standard)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pos_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    starts.append(_project_to_sphere(pos_fib))
    
    # 2. Fibonacci spiral with small perturbation
    rng = np.random.default_rng(42)
    pos_fib_pert = pos_fib + rng.normal(0, 0.01, size=(N, 3))
    starts.append(_project_to_sphere(pos_fib_pert))
    
    # 3. Random uniform points
    for _ in range(3):
        pos_rand = rng.normal(0, 1, size=(N, 3))
        starts.append(_project_to_sphere(pos_rand))
        
    # 4. More random ones for diversity
    for _ in range(2):
        pos_rand = rng.normal(0, 1, size=(N, 3))
        starts.append(_project_to_sphere(pos_rand))

    best_pos = starts[0]
    best_energy = float('inf')
    
    # Allocate time budget to each start
    # We have TIME_LIMIT seconds. 
    # Let's give 60% of time to the first few most promising starts (Fibonacci)
    # and the rest to random ones.
    
    total_opt_time = TIME_LIMIT - 10.0 # 10s buffer for overhead and final checks
    time_per_start = [
        total_opt_time * 0.30, # Fib
        total_opt_time * 0.20, # Fib Pert
        total_opt_time * 0.10, # Rand 1
        total_opt_time * 0.10, # Rand 2
        total_opt_time * 0.10, # Rand 3
        total_opt_time * 0.10, # Rand 4
        total_opt_time * 0.10, # Rand 5
    ]
    
    for idx, start_pos in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 5:
            break
            
        # Ensure we don't exceed total limit
        alloc_time = min(time_per_start[idx], TIME_LIMIT - 5 - elapsed)
        if alloc_time < 1.0:
            continue
            
        pos, energy = _optimize(start_pos, alloc_time)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
            
    # Final validation and projection
    best_pos = _project_to_sphere(best_pos)
    
    if not np.all(np.isfinite(best_pos)):
        # Fallback to Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        best_pos = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        best_pos = _project_to_sphere(best_pos)
        
    # Check minimum distance
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        # Perturb and retry quickly
        rng = np.random.default_rng(123)
        perturbed = best_pos + rng.normal(0, 0.01, size=(N, 3))
        perturbed = _project_to_sphere(perturbed)
        # Quick optimization
        pos, energy = _optimize(perturbed, 5.0)
        if energy < best_energy:
            best_pos = pos
            best_energy = energy
            best_pos = _project_to_sphere(best_pos)

    return best_pos
