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
    norms = np.where(norms < 1e-12, 1.0, norms)
    x = x / norms
    
    # Pairwise differences using matrix operations for speed
    dot = x @ x.T
    dist_sq = 2.0 - 2.0 * dot
    np.maximum(dist_sq, 1e-30, out=dist_sq)
    np.fill_diagonal(dist_sq, 1.0)
    
    # Mask for i < j to compute energy
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d_pairs = np.sqrt(dist_sq[mask])
    energy = -np.sum(np.log(d_pairs))
    
    # Gradient computation
    weights = 1.0 / dist_sq
    np.fill_diagonal(weights, 0.0)
    
    w_sum = np.sum(weights, axis=1)
    w_x = weights @ x
    
    grad_euclid = w_x - x * w_sum[:, np.newaxis]
    
    # Project to tangent space
    g_dot_x = np.sum(grad_euclid * x, axis=1, keepdims=True)
    grad_tangent = grad_euclid - g_dot_x * x
    
    return energy, grad_tangent.flatten()

def _energy_eval(x):
    """
    High precision energy calculation for final evaluation.
    """
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    dot = x @ x.T
    dist_sq = 2.0 - 2.0 * dot
    np.maximum(dist_sq, 1e-30, out=dist_sq)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d_pairs = np.sqrt(dist_sq[mask])
    return -math.fsum(np.log(d_pairs).tolist())

def _riemannian_grad_descent(x0, max_iter=3000, time_limit=20.0, deadline=1e9):
    """
    Riemannian gradient descent with adaptive step size and backtracking.
    """
    start_time = time.time()
    x = x0.copy()
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    
    lr = 0.08
    best_energy = float('inf')
    best_x = x.copy()
    patience = 0
    max_patience = 500
    
    for it in range(max_iter):
        if time.time() > start_time + time_limit:
            break
        if time.time() > deadline - 30.0:
            break
        
        energy, grad = _objective_and_grad(x.flatten())
        
        if energy < best_energy:
            best_energy = energy
            best_x = x.copy()
            patience = 0
        else:
            patience += 1
        
        if patience > max_patience:
            break
        
        # Normalize gradient
        g_norm = np.linalg.norm(grad)
        if g_norm < 1e-14:
            break
        grad = grad / g_norm
        
        # Backtracking line search
        alpha = lr
        success = False
        for _ in range(10):
            x_new = x - alpha * grad.reshape(N, 3)
            x_new = x_new / np.linalg.norm(x_new, axis=1, keepdims=True)
            
            e_new, _ = _objective_and_grad(x_new.flatten())
            if e_new < energy:
                x = x_new
                success = True
                break
            alpha *= 0.5
        
        if not success:
            alpha = 0.01
            x_new = x - alpha * grad.reshape(N, 3)
            x_new = x_new / np.linalg.norm(x_new, axis=1, keepdims=True)
            e_new, _ = _objective_and_grad(x_new.flatten())
            if e_new < energy:
                x = x_new
            else:
                break
        
        # Adapt learning rate
        if success and alpha > lr * 0.5:
            lr = min(lr * 1.05, 0.15)
        else:
            lr = max(lr * 0.9, 0.02)
    
    return best_x, best_energy

def _optimize_l_bfgs(x0, time_limit=15.0):
    """
    Optimize using L-BFGS-B with moderate tolerances.
    """
    x0_flat = x0.flatten()
    
    res = minimize(
        _objective_and_grad, 
        x0_flat, 
        method='L-BFGS-B', 
        jac=True,
        options={
            'maxiter': 2000,
            'ftol': 1e-18,
            'gtol': 1e-12,
            'maxls': 40,
            'maxcor': 30,
            'disp': False
        }
    )
    
    x_opt = res.x.reshape(N, 3)
    norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    x_opt = x_opt / norms
    
    return x_opt, res.fun

def _optimize_bfgs_tight(x0, time_limit=25.0):
    """
    High-precision polish using BFGS with very tight tolerances.
    """
    x0_flat = x0.flatten()
    
    res = minimize(
        _objective_and_grad, 
        x0_flat, 
        method='BFGS', 
        jac=True,
        options={
            'maxiter': 1000,
            'gtol': 1e-14,
            'disp': False
        }
    )
    
    x_opt = res.x.reshape(N, 3)
    norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    x_opt = x_opt / norms
    
    return x_opt, res.fun

def _polish_newton(x0, time_limit=20.0):
    """
    Ultra-tight L-BFGS-B polish for final steps.
    """
    x0_flat = x0.flatten()
    
    res = minimize(
        _objective_and_grad, 
        x0_flat, 
        method='L-BFGS-B', 
        jac=True,
        options={
            'maxiter': 1000,
            'ftol': 1e-20,
            'gtol': 1e-16,
            'maxls': 50,
            'maxcor': 30,
            'disp': False
        }
    )
    
    x_opt = res.x.reshape(N, 3)
    norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    x_opt = x_opt / norms
    
    return x_opt, res.fun

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_x = None
    best_e = float('inf')
    
    # Generate diverse initializations
    inits = []
    
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    inits.append(x_fib)
    
    # 2. Perturbed Fibonacci Spirals (more scales)
    for scale in [0.03, 0.08, 0.15, 0.3, 0.6, 1.0]:
        noise = rng.normal(0, scale, (N, 3))
        x_pert = x_fib + noise
        x_pert /= np.linalg.norm(x_pert, axis=1, keepdims=True)
        inits.append(x_pert)
        
    # 3. Random Initializations
    for _ in range(8):
        x_rand = rng.normal(size=(N, 3))
        x_rand /= np.linalg.norm(x_rand, axis=1, keepdims=True)
        inits.append(x_rand)
    
    # 4. Antipodal pairs + jitter
    for _ in range(3):
        x_base = rng.normal(size=(N//2, 3))
        x_base /= np.linalg.norm(x_base, axis=1, keepdims=True)
        x_anti = np.vstack([x_base, -x_base])
        jitter = rng.normal(0, 0.05, (N, 3))
        x_anti = x_anti + jitter
        x_anti /= np.linalg.norm(x_anti, axis=1, keepdims=True)
        inits.append(x_anti)
    
    # 5. Clustered initialization
    for _ in range(2):
        centers = rng.normal(size=(15, 3))
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        x_clustered = np.array([centers[i % 15] + rng.normal(0, 0.1, 3) for i in range(N)])
        x_clustered /= np.linalg.norm(x_clustered, axis=1, keepdims=True)
        inits.append(x_clustered)
    
    # Phase 1: L-BFGS-B optimization on each initialization
    for idx, x0 in enumerate(inits):
        if time.time() > deadline - 60.0:
            break
            
        try:
            x_opt, e_opt = _optimize_l_bfgs(x0, time_limit=12.0)
            e_precise = _energy_eval(x_opt)
            
            if e_precise < best_e:
                best_e = e_precise
                best_x = x_opt.copy()
        except Exception:
            continue
            
    if best_x is None:
        best_x = inits[0]
        best_e = _energy_eval(best_x)
        
    # Phase 2: Riemannian gradient descent polish
    if time.time() < deadline - 45.0:
        try:
            x_rg, e_rg = _riemannian_grad_descent(best_x, max_iter=3000, time_limit=25.0, deadline=deadline)
            e_rg_precise = _energy_eval(x_rg)
            if e_rg_precise < best_e:
                best_x = x_rg
                best_e = e_rg_precise
        except Exception:
            pass
        
    # Phase 3: BFGS tight polish
    if time.time() < deadline - 25.0:
        try:
            x_final3, e_final3 = _optimize_bfgs_tight(best_x, time_limit=20.0)
            e_final3_precise = _energy_eval(x_final3)
            if e_final3_precise < best_e:
                best_x = x_final3
                best_e = e_final3_precise
        except Exception:
            pass
        
    # Phase 4: Ultra-tight L-BFGS-B polish (Newton-like)
    if time.time() < deadline - 5.0:
        try:
            x_final4, e_final4 = _polish_newton(best_x, time_limit=15.0)
            e_final4_precise = _energy_eval(x_final4)
            if e_final4_precise < best_e:
                best_x = x_final4
                best_e = e_final4_precise
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
