import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def _objective_and_grad(x_flat):
    """
    Calculate the logarithmic energy and its Riemannian gradient (projected to tangent space)
    for points on the unit sphere.
    x_flat: (3*N,) array
    Returns: energy (float), gradient (3*N,) array
    """
    x = x_flat.reshape(N, 3)
    
    # Normalize points to ensure they are on the sphere
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms < 1e-12, 1.0, norms)
    x = x / norms
    
    # Pairwise differences
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    
    # Mask for i < j to compute energy
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d_pairs = np.sqrt(dist_sq[mask] + 1e-30)
    
    # Energy: sum log(1/d) = -sum log(d)
    # Using np.sum for speed in optimizer loop
    energy = -np.sum(np.log(d_pairs))
    
    # Gradient
    # Euclidean gradient of E w.r.t x_i is sum_{j != i} (x_j - x_i) / ||x_i - x_j||^2
    # Note: d/dx_i log(||x_i - x_j||) = (x_i - x_j)/||x_i - x_j||^2
    # So d/dx_i (-log(||x_i - x_j||)) = -(x_i - x_j)/||x_i - x_j||^2 = (x_j - x_i)/||x_i - x_j||^2
    # This matches the formula used.
    
    weights = 1.0 / (dist_sq + 1e-30)
    np.fill_diagonal(weights, 0.0)
    
    w_sum = np.sum(weights, axis=1)
    w_x = weights @ x
    
    grad_euclid = w_x - x * w_sum[:, np.newaxis]
    
    # Project to tangent space
    g_dot_x = np.sum(grad_euclid * x, axis=1, keepdims=True)
    grad_tangent = grad_euclid - g_dot_x * x
    
    return energy, grad_tangent.flatten()

def _optimize_scipy(x0, max_iter=2000):
    """
    Optimize using L-BFGS-B.
    """
    x0_flat = x0.flatten()
    
    res = minimize(
        _objective_and_grad, 
        x0_flat, 
        method='L-BFGS-B', 
        jac=True,
        options={
            'maxiter': max_iter,
            'ftol': 1e-18,
            'gtol': 1e-14,
            'maxls': 100,
            'maxcor': 30,
            'disp': False
        }
    )
    
    x_opt = res.x.reshape(N, 3)
    
    # Normalize again to ensure validity
    norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    x_opt = x_opt / norms
    
    return x_opt, res.fun

def _energy_eval(x):
    """
    High precision energy calculation for final evaluation.
    """
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d_pairs = np.sqrt(dist_sq[mask] + 1e-30)
    return -math.fsum(np.log(d_pairs).tolist())

def construct():
    start_time = time.time()
    # Evaluator kills at 150s. Leave a safety margin of 10s.
    deadline = start_time + 140.0
    
    rng = np.random.default_rng(42)
    
    best_x = None
    best_e = float('inf')
    
    # 1. Generate Initializations
    inits = []
    
    # Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    inits.append(x_fib)
    
    # Perturbed Fibonacci Spirals
    for scale in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        noise = rng.normal(0, scale, (N, 3))
        x_pert = x_fib + noise
        x_pert /= np.linalg.norm(x_pert, axis=1, keepdims=True)
        inits.append(x_pert)
        
    # Random Initializations
    # Add more random starts to explore the landscape
    for _ in range(20):
        x_rand = rng.normal(size=(N, 3))
        x_rand /= np.linalg.norm(x_rand, axis=1, keepdims=True)
        inits.append(x_rand)
        
    # Optimize each initialization
    # Allocate time dynamically
    # We have 30 initializations.
    # Let's try to give each a fair share, but prioritize the first few which are often better.
    # Or just run them all quickly and then polish the best.
    
    for idx, x0 in enumerate(inits):
        if time.time() > deadline - 15.0:
            break
            
        try:
            # Allocate time: first few get more time, later ones less
            # Or just a fixed time per run, e.g., 3 seconds
            # 30 * 3s = 90s. Leaves 35s for polish.
            t_alloc = 3.0
            
            x_opt, e_opt = _optimize_scipy(x0, max_iter=500) # Limit iterations to fit in time
            
            # Verify with high precision energy
            e_precise = _energy_eval(x_opt)
            
            if e_precise < best_e:
                best_e = e_precise
                best_x = x_opt
        except Exception:
            continue
            
    if best_x is None:
        best_x = x_fib
        best_e = _energy_eval(best_x)
        
    # Final polish on the best solution
    # Use remaining time for a very tight optimization
    if time.time() < deadline - 5.0:
        try:
            # Run with max iterations and tight tolerance
            x_final, e_final = _optimize_scipy(best_x, max_iter=5000)
            e_final_precise = _energy_eval(x_final)
            if e_final_precise < best_e:
                best_x = x_final
                best_e = e_final_precise
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
