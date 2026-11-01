"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

def _energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|)
    and its gradient w.r.t. the Cartesian coordinates.
    E = -sum_{i<j} log(|x_i - x_j|)
    dE/dx_i = - sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist2 = np.sum(diff * diff, axis=2)  # (n, n)
    # Avoid division by zero on the diagonal
    np.fill_diagonal(dist2, 1.0)
    
    dist = np.sqrt(dist2)
    
    # Energy: sum over i<j of log(1/dist[i,j]) = -sum over i<j log(dist[i,j])
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: dE/dx_i = - sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    weight = np.zeros_like(dist2)
    np.fill_diagonal(weight, 0.0)
    weight = 1.0 / dist2
    
    grad = -np.einsum('ij,ijk->ik', weight, diff)
    
    return energy, grad

def _sphere_energy_and_grad(x_flat):
    """
    Objective and gradient for scipy.optimize.minimize.
    x_flat is a 3*N array of coordinates.
    We normalize each point to the sphere inside this function.
    """
    N_local = len(x_flat) // 3
    pts = x_flat.reshape(N_local, 3)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    
    energy, grad = _energy_and_grad(pts)
    
    # Project gradient onto tangent space for the effective gradient
    # For f(x) = E(x/||x||), the gradient w.r.t x is (I - x x^T/||x||^2) * grad_E
    # Since pts is already normalized (||pts||=1), the projection is:
    # grad_T = grad - (grad . pts) * pts
    proj = np.einsum('ij,ij->i', grad, pts)
    grad_tangent = grad - proj[:, np.newaxis] * pts
    
    return energy, grad_tangent.flatten()

def _retract(pts, grad_tangent, step_size):
    """
    Retraction on the sphere: move along tangent direction and re-normalize.
    """
    new_pts = pts - step_size * grad_tangent
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    new_pts = new_pts / np.maximum(norms, 1e-15)
    return new_pts

def _lbfgs_sphere(x0, deadline, maxiter=5000):
    """
    Run L-BFGS-B on the sphere with time guard.
    """
    start_time = time.time()
    
    best_x = x0.copy()
    best_f, _ = _sphere_energy_and_grad(x0)
    
    def callback(xk):
        nonlocal best_x, best_f
        if time.time() > deadline:
            return True
        f_k, _ = _sphere_energy_and_grad(xk)
        if f_k < best_f:
            best_f = f_k
            best_x = xk.copy()
        return False
    
    try:
        res = minimize(
            _sphere_energy_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={
                'maxiter': maxiter,
                'ftol': 1e-14,
                'gtol': 1e-12,
                'maxfun': 10000
            },
            callback=callback
        )
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x.copy()
    except Exception:
        pass
    
    return best_x, best_f

def _polish(pts, deadline, max_iter=200):
    """
    High-precision polish using gradient descent with backtracking line search
    on the tangent space.
    """
    start_time = time.time()
    f, g = _energy_and_grad(pts)
    proj = np.einsum('ij,ij->i', g, pts)
    g_tangent = g - proj[:, np.newaxis] * pts
    
    lr = 0.05
    for it in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        
        new_pts = _retract(pts, g_tangent, lr)
        new_f, new_g = _energy_and_grad(new_pts)
        new_proj = np.einsum('ij,ij->i', new_g, new_pts)
        new_g_tangent = new_g - new_proj[:, np.newaxis] * new_pts
        
        if new_f < f:
            pts = new_pts
            f = new_f
            g = new_g
            g_tangent = new_g_tangent
            lr = min(lr * 1.1, 0.1)
        else:
            lr *= 0.5
            if lr < 1e-12:
                break
    
    return pts, f

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Generate multiple diverse initializations
    initializations = []
    
    # 1. Fibonacci spiral (golden angle)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initializations.append(fib_pts)
    
    # 2. Random points (seeded for reproducibility)
    rng = np.random.default_rng(42)
    for seed_offset in range(5):
        rng_local = np.random.default_rng(42 + seed_offset)
        random_pts = rng_local.standard_normal((N, 3))
        norms = np.linalg.norm(random_pts, axis=1, keepdims=True)
        random_pts = random_pts / np.maximum(norms, 1e-15)
        initializations.append(random_pts)
    
    # 3. Perturbed Fibonacci
    rng_pert = np.random.default_rng(123)
    for pert_scale in [0.01, 0.05, 0.1]:
        perturbed = fib_pts + rng_pert.standard_normal((N, 3)) * pert_scale
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        perturbed = perturbed / np.maximum(norms, 1e-15)
        initializations.append(perturbed)
    
    # 4. Regular icosahedral-like base with perturbations
    # Use a few more structured initializations
    for angle_shift in [0.1, 0.2, 0.3]:
        i2 = np.arange(N) + 0.5
        z2 = 1.0 - 2.0 * i2 / N
        phi2 = np.pi * (1.0 + np.sqrt(5.0)) * i2 + angle_shift
        r2 = np.sqrt(1.0 - z2 * z2)
        shifted_pts = np.column_stack([r2 * np.cos(phi2), r2 * np.sin(phi2), z2])
        initializations.append(shifted_pts)
    
    # Optimize each initialization and keep the best
    best_x = None
    best_f = float('inf')
    
    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 10.0:
            break
        
        x0 = init_pts.flatten()
        
        # Quick check: how much time do we have?
        remaining = deadline - time.time()
        if remaining < 10.0:
            break
        
        # Allocate time proportionally: earlier initializations get more time
        # but we want to try all of them
        maxiter_for_this = min(5000, int(remaining * 20))
        if maxiter_for_this < 100:
            maxiter_for_this = 100
        
        x_opt, f_opt = _lbfgs_sphere(x0, deadline, maxiter=maxiter_for_this)
        
        if f_opt < best_f:
            best_f = f_opt
            best_x = x_opt.copy()
    
    if best_x is None:
        # Fallback to Fibonacci
        best_x = fib_pts.flatten()
        best_f, _ = _sphere_energy_and_grad(best_x)
    
    # Final high-precision polish on the best result
    if time.time() < deadline - 5.0:
        pts = best_x.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / np.maximum(norms, 1e-15)
        
        pts, f = _polish(pts, deadline, max_iter=300)
        best_x = pts.flatten()
        best_f = f
    
    # Ensure points are on the sphere
    pts_opt = best_x.reshape(N, 3)
    norms = np.linalg.norm(pts_opt, axis=1, keepdims=True)
    pts_opt = pts_opt / np.maximum(norms, 1e-15)
    
    return pts_opt
# EVOLVE-BLOCK-END
