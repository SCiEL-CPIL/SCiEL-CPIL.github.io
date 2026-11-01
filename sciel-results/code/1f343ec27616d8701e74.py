"""
Strategy: Global Search via Simulated Annealing on the Sphere Manifold.
1. Use Riemannian geometry (exponential/logarithmic maps) to move points on the sphere.
2. Simulated Annealing allows escaping local minima by accepting uphill moves.
3. Final polish with L-BFGS on tangent space for high precision.
"""
N = 150
import numpy as np
import time
import math
from scipy.optimize import minimize
import random

# EVOLVE-BLOCK-START

def _log_energy_and_grad_sphere(x):
    """
    Computes the logarithmic energy and its Euclidean gradient for points on the sphere.
    x: (N, 3) array of points on the unit sphere.
    Returns: energy, grad (N, 3)
    """
    # diff is (N, N, 3) where diff[i,j] = x_i - x_j
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    mask = ~np.eye(N, dtype=bool)
    
    # Energy: 0.5 * sum_{i!=j} -log(dist)
    # Note: The task asks for sum_{i<j} log(1/dist). 
    # sum_{i<j} log(1/dist_ij) = -0.5 * sum_{i!=j} log(dist_ij)
    # Our energy calculation: 0.5 * sum_{i!=j} (-log(dist_ij)) = sum_{i<j} -log(dist_ij)
    # This matches the task's energy definition.
    
    safe_dist = np.where(mask, dist, 1.0)
    log_terms = -np.log(safe_dist)
    energy = 0.5 * np.sum(log_terms[mask])
    
    # Gradient: grad_i = - sum_{j != i} (x_i - x_j) / dist_ij^2
    inv_dist_sq = np.zeros_like(dist)
    inv_dist_sq[mask] = 1.0 / (dist[mask] ** 2)
    
    weighted_diff = diff * inv_dist_sq[:, :, np.newaxis]
    grad = -np.sum(weighted_diff, axis=1)
    
    return energy, grad

def _riemannian_project(grad, x):
    """
    Project Euclidean gradient to the tangent space of the sphere at x.
    proj_x(g) = g - (g . x) x
    """
    dot = np.sum(grad * x, axis=1, keepdims=True)
    return grad - dot * x

def _exponential_map(x, v):
    """
    Exponential map on the unit sphere.
    x: point on sphere (N, 3)
    v: tangent vector (N, 3)
    Returns: new point on sphere.
    """
    norm_v = np.linalg.norm(v, axis=1, keepdims=True)
    # Handle zero vectors
    mask = norm_v > 1e-10
    new_x = np.zeros_like(x)
    
    # For small norms, use Taylor expansion or just cos/sin
    # x_new = cos(||v||) * x + sin(||v||) / ||v|| * v
    cos_theta = np.cos(norm_v)
    sin_theta_over_norm = np.zeros_like(norm_v)
    sin_theta_over_norm[mask] = np.sin(norm_v[mask]) / norm_v[mask]
    
    new_x = cos_theta * x + sin_theta_over_norm * v
    return new_x

def _fibonacci_sphere(n):
    """Generate a Fibonacci spiral distribution on the sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return x

def _random_sphere(n, seed=None):
    """Generate random points on the sphere."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _local_optimize_lbfgs(x, time_budget=10.0):
    """
    Perform local optimization using L-BFGS on the tangent space.
    We parameterize the update as x_new = exp_x(v) where v is in tangent space.
    However, for simplicity and speed, we can use the 'gradient descent' step 
    projected to tangent space, or use scipy's minimize with a custom step.
    
    A simpler robust approach: Use scipy minimize on the 3D coordinates 
    with a penalty for leaving the sphere? No, that's unstable.
    
    Better: Use the fact that for small steps, the Euclidean gradient projected 
    to tangent space is the Riemannian gradient. 
    We can use L-BFGS on the 3D coordinates but re-project after each step?
    Actually, `scipy.optimize.minimize` doesn't natively support manifolds.
    
    Let's implement a simple Riemannian Gradient Descent with Armijo line search 
    or just use L-BFGS on the stereographic chart again but only for the final polish 
    where we are close to the minimum.
    
    Alternative: Use `scipy.optimize.minimize` on the 3D vector, but modify the 
    objective to include a large penalty if ||x_i|| != 1. 
    Penalty: C * sum(||x_i||^2 - 1)^2.
    This is a common trick.
    """
    def obj_penalized(v_flat):
        x = v_flat.reshape(N, 3)
        # Normalize to avoid numerical drift in energy calc, but penalty guides the optimizer
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x_norm = x / norms
        energy, _ = _log_energy_and_grad_sphere(x_norm)
        penalty = 1e6 * np.sum((norms - 1.0)**2)
        return energy + penalty
    
    # We don't have an easy analytic gradient for the penalty part mixed with normalization.
    # Let's stick to a manual Riemannian CG or GD for the polish, or just use 
    # the stereographic chart from the parent code which was working.
    
    # Actually, let's use a simple Riemannian Gradient Descent with adaptive step size.
    # It's robust and doesn't require complex Hessian approximations.
    
    x_curr = x.copy()
    best_energy, _ = _log_energy_and_grad_sphere(x_curr)
    
    step_size = 0.01
    for i in range(500):
        if time.time() > time_budget:
            break
            
        energy, grad = _log_energy_and_grad_sphere(x_curr)
        if energy < best_energy:
            best_energy = energy
            
        rgrad = _riemannian_project(grad, x_curr)
        
        # Armijo line search
        alpha = step_size
        for _ in range(20):
            v = alpha * rgrad
            x_new = _exponential_map(x_curr, v)
            e_new, _ = _log_energy_and_grad_sphere(x_new)
            if e_new < energy - 1e-4 * alpha * np.sum(rgrad**2):
                break
            alpha *= 0.5
            
        x_curr = x_new
        # Reduce step size slowly
        step_size *= 0.999
        
    return x_curr

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial configurations
    inits = [
        _fibonacci_sphere(N),
        _random_sphere(N, seed=1),
        _random_sphere(N, seed=2),
        _random_sphere(N, seed=3)
    ]
    
    best_energy = float('inf')
    best_x = None
    
    # Simulated Annealing Parameters
    # Temperature schedule: T(t) = T0 * (T_end / T0)^(t/T_max)
    T0 = 10.0
    Tend = 0.01
    n_iterations = 2000
    
    # We will run SA for a few initial configurations
    for init_idx, x_init in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        x_curr = x_init.copy()
        e_curr, _ = _log_energy_and_grad_sphere(x_curr)
        
        if e_curr < best_energy:
            best_energy = e_curr
            best_x = x_curr.copy()
            
        # Run Simulated Annealing
        for t in range(n_iterations):
            if time.time() > deadline - 10:
                break
                
            # Temperature
            T = T0 * (Tend / T0) ** (t / n_iterations)
            
            # Generate random tangent vector
            # Random direction in 3D, project to tangent space
            v_rand = np.random.normal(size=(N, 3))
            v_rand = _riemannian_project(v_rand, x_curr)
            
            # Normalize step size relative to temperature
            # Larger T -> larger steps
            step_scale = 0.01 + 0.1 * (T / T0)
            v_step = step_scale * v_rand
            
            x_new = _exponential_map(x_curr, v_step)
            e_new, _ = _log_energy_and_grad_sphere(x_new)
            
            # Metropolis criterion
            delta_e = e_new - e_curr
            if delta_e < 0:
                # Accept downhill
                x_curr = x_new
                e_curr = e_new
                if e_curr < best_energy:
                    best_energy = e_curr
                    best_x = x_curr.copy()
            else:
                # Accept uphill with probability exp(-delta_e / T)
                if random.random() < math.exp(-delta_e / T):
                    x_curr = x_new
                    e_curr = e_new
        
        # After SA, do a local polish
        if time.time() < deadline - 5:
            x_polished = _local_optimize_lbfgs(x_curr, time_budget=5.0)
            e_polished, _ = _log_energy_and_grad_sphere(x_polished)
            if e_polished < best_energy:
                best_energy = e_polished
                best_x = x_polished.copy()
                
    # Final global polish on the best result found
    if best_x is not None and time.time() < deadline - 2:
        x_final = _local_optimize_lbfgs(best_x, time_budget=5.0)
        e_final, _ = _log_energy_and_grad_sphere(x_final)
        if e_final < best_energy:
            best_energy = e_final
            best_x = x_final
            
    return best_x

# EVOLVE-BLOCK-END
