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
    np.fill_diagonal(dist2, 1.0)
    
    dist = np.sqrt(dist2)
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    weight = np.zeros_like(dist2)
    np.fill_diagonal(weight, 0.0)
    weight = 1.0 / dist2
    
    grad = -np.einsum('ij,ijk->ik', weight, diff)
    
    return energy, grad

def _sphere_energy_and_grad(x_flat):
    """
    Objective and gradient for scipy.optimize.minimize.
    x_flat is a 3*N array of coordinates.
    """
    N_local = len(x_flat) // 3
    pts = x_flat.reshape(N_local, 3)
    
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    
    energy, grad = _energy_and_grad(pts)
    
    proj = np.einsum('ij,ij->i', grad, pts)
    grad_tangent = grad - proj[:, np.newaxis] * pts
    
    return energy, grad_tangent.flatten()

def _retract(pts, direction, step_size):
    """
    Retraction on the sphere: move along tangent direction and re-normalize.
    """
    new_pts = pts + step_size * direction
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    new_pts = new_pts / np.maximum(norms, 1e-15)
    return new_pts

def _lbfgs_sphere(x0, deadline, maxiter=8000):
    """
    Run L-BFGS-B on the sphere with time guard and tight tolerances.
    """
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
                'ftol': 1e-16,
                'gtol': 1e-14,
                'maxfun': 20000
            },
            callback=callback
        )
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x.copy()
    except Exception:
        pass
    
    return best_x, best_f

def _polish(pts, deadline, max_iter=800, lr_init=0.05, lr_max=0.15):
    """
    High-precision polish using gradient descent with backtracking line search
    on the tangent space. Uses adaptive learning rate.
    """
    f, g = _energy_and_grad(pts)
    proj = np.einsum('ij,ij->i', g, pts)
    g_tangent = g - proj[:, np.newaxis] * pts
    
    lr = lr_init
    for it in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        
        new_pts = _retract(pts, -g_tangent, lr)
        new_f, new_g = _energy_and_grad(new_pts)
        new_proj = np.einsum('ij,ij->i', new_g, new_pts)
        new_g_tangent = new_g - new_proj[:, np.newaxis] * new_pts
        
        if new_f < f:
            pts = new_pts
            f = new_f
            g = new_g
            g_tangent = new_g_tangent
            lr = min(lr * 1.15, lr_max)
        else:
            lr *= 0.5
            if lr < 1e-14:
                break
    
    return pts, f

def _ultra_polish(pts, deadline, max_iter=3000):
    """
    Ultra-precise Newton-style polish with very small steps and tight Armijo.
    """
    f, g = _energy_and_grad(pts)
    proj = np.einsum('ij,ij->i', g, pts)
    g_t = g - proj[:, np.newaxis] * pts
    
    # Estimate local curvature scale from gradient magnitude
    g_norm = np.linalg.norm(g_t)
    if g_norm < 1e-15:
        return pts, f
    
    # Start with a small step relative to gradient scale
    alpha = 0.01
    for it in range(max_iter):
        if time.time() > deadline - 1.0:
            break
        
        # Armijo backtracking
        sufficient = 1e-18 * alpha * np.sum(g_t * (-g_t))
        cand = _retract(pts, -g_t, alpha)
        cand_f, cand_g = _energy_and_grad(cand)
        cand_proj = np.einsum('ij,ij->i', cand_g, cand)
        cand_g_t = cand_g - cand_proj[:, np.newaxis] * cand
        
        if cand_f <= f - sufficient:
            pts = cand
            f = cand_f
            g = cand_g
            g_t = cand_g_t
            # Slightly increase step if we made good progress
            if it % 10 == 0:
                alpha = min(alpha * 1.05, 0.05)
        else:
            alpha *= 0.5
            if alpha < 1e-18:
                break
        
        # Check convergence
        if np.linalg.norm(g_t) < 1e-15:
            break
    
    return pts, f

def _lbfgs_sphere_custom(pts, max_iter=500, tol=1e-12, deadline=None, m=25):
    """
    Custom L-BFGS on the sphere using Riemannian gradient and retraction.
    """
    n = pts.shape[0]
    x = pts.copy()
    f, g = _energy_and_grad(x)
    
    proj = np.einsum('ij,ij->i', g, x)
    g_t = g - proj[:, np.newaxis] * x
    
    s_list = []
    y_list = []
    
    for k in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        if np.linalg.norm(g_t) < tol:
            break
        
        if len(s_list) > 0:
            rhos = []
            for s, y in zip(s_list, y_list):
                sy = np.sum(s * y)
                rhos.append(1.0 / sy if abs(sy) > 1e-15 else 0.0)
            
            q = g_t.copy()
            alphas = []
            for i in range(len(s_list) - 1, -1, -1):
                s_i = s_list[i]
                y_i = y_list[i]
                rho_i = rhos[i]
                alpha_i = rho_i * np.sum(s_i * q)
                alphas.append(alpha_i)
                q = q - alpha_i * y_i
            
            gamma = 1.0
            r = -gamma * q
            
            for i in range(len(s_list)):
                s_i = s_list[i]
                y_i = y_list[i]
                rho_i = rhos[i]
                beta_i = rho_i * np.sum(y_i * r)
                r = r + (alphas[i] - beta_i) * s_i
            
            direction = r
        else:
            direction = -g_t
        
        # Line search with backtracking
        step_size = 1.0
        new_x = x
        new_f = f
        new_g = g
        new_g_t = g_t
        for _ in range(50):
            candidate = _retract(x, direction, step_size)
            new_f, new_g = _energy_and_grad(candidate)
            new_proj = np.einsum('ij,ij->i', new_g, candidate)
            new_g_t = new_g - new_proj[:, np.newaxis] * candidate
            
            if new_f <= f + 1e-5 * step_size * np.sum(g_t * direction):
                new_x = candidate
                break
            step_size *= 0.5
        else:
            new_x = candidate
        
        s_k = new_x - x
        y_k = new_g_t - g_t
        sy = np.sum(s_k * y_k)
        if sy > 1e-12:
            s_list.append(s_k)
            y_list.append(y_k)
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)
        
        x = new_x
        f = new_f
        g = new_g
        proj = np.einsum('ij,ij->i', g, x)
        g_t = g - proj[:, np.newaxis] * x
    
    return x, f

def _fibonacci_spiral():
    """Generate Fibonacci spiral initial points."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=None):
    """Generate n random points on the sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _deduplicate(pts_list, threshold=0.3):
    """Remove initializations that are too similar to already-kept ones."""
    kept = []
    for pts in pts_list:
        too_close = False
        for kp in kept:
            # Compute min pairwise distance
            diff = pts[:, np.newaxis, :] - kp[np.newaxis, :, :]
            dmin = np.min(np.linalg.norm(diff, axis=2))
            if dmin < threshold:
                too_close = True
                break
        if not too_close:
            kept.append(pts)
    return kept

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Generate diverse initial configurations
    inits = []
    inits.append(_fibonacci_spiral())
    
    # Add several random starts with different seeds
    for seed in [42, 123, 456, 789, 101112]:
        inits.append(_random_sphere(N, seed=seed))
    
    # Deduplicate
    inits = _deduplicate(inits, threshold=0.2)
    
    best_pts = None
    best_f = np.inf
    
    # Optimize each initialization
    for idx, init in enumerate(inits):
        remaining = deadline - time.time()
        if remaining < 10.0:
            break
        
        # Allocate time proportionally, but ensure at least some time for each
        time_per_init = min(remaining / max(1, len(inits) - idx), 30.0)
        init_deadline = time.time() + time_per_init
        
        # Run L-BFGS-B
        pts_opt, f_opt = _lbfgs_sphere(init.flatten(), init_deadline, maxiter=3000)
        pts_opt = pts_opt.reshape(N, 3)
        
        # Polish if time permits
        if time.time() < init_deadline - 3.0:
            pts_opt, f_opt = _polish(pts_opt, init_deadline, max_iter=300)
        
        # Ultra-polish
        if time.time() < init_deadline - 1.0:
            pts_opt, f_opt = _ultra_polish(pts_opt, init_deadline, max_iter=1000)
        
        if f_opt < best_f:
            best_f = f_opt
            best_pts = pts_opt.copy()
    
    # Final refinement on the best configuration
    if best_pts is not None and time.time() < deadline - 5.0:
        # Custom L-BFGS for additional refinement
        remaining = deadline - time.time()
        if remaining > 5.0:
            pts_ref, f_ref = _lbfgs_sphere_custom(best_pts, max_iter=200, tol=1e-14, 
                                                   deadline=deadline - 3.0)
            if f_ref < best_f:
                best_pts = pts_ref
                best_f = f_ref
        
        # Final ultra-polish
        if time.time() < deadline - 2.0:
            best_pts, best_f = _ultra_polish(best_pts, deadline, max_iter=2000)
    
    # Ensure points are on the sphere and valid
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / np.maximum(norms, 1e-15)
    
    return best_pts
# EVOLVE-BLOCK-END
