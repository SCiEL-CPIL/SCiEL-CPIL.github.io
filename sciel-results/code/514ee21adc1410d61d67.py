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

def _retract(pts):
    """Project points back onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _lbfgs_sphere(x0, deadline):
    """Run L-BFGS-B with manifold-aware gradient projection."""
    n = len(x0) // 3
    
    def obj_and_grad(x_flat):
        pts = x_flat.reshape(n, 3)
        pts = _retract(pts)
        energy, grad = _energy_and_grad(pts)
        # Project gradient to tangent space for correct unconstrained optimization
        proj = np.einsum('ij,ij->i', grad, pts)
        grad_tangent = grad - proj[:, np.newaxis] * pts
        return energy, grad_tangent.flatten()
    
    best_x = x0.copy()
    best_f, _ = obj_and_grad(x0)
    
    def callback(xk):
        nonlocal best_x, best_f
        if time.time() > deadline:
            return True
        f_k, _ = obj_and_grad(xk)
        if f_k < best_f:
            best_f = f_k
            best_x = xk.copy()
        return False
    
    try:
        res = minimize(
            obj_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 10000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 20000},
            callback=callback
        )
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x
    except Exception:
        pass
    
    return _retract(best_x.reshape(n, 3))

def _polish(pts, deadline):
    """High-precision manifold gradient descent polish with adaptive step size."""
    n = pts.shape[0]
    lr = 1e-2
    f_curr, g_curr = _energy_and_grad(pts)
    
    for it in range(500):
        if time.time() > deadline - 2.0:
            break
        
        # Project gradient to tangent space
        proj = np.einsum('ij,ij->i', g_curr, pts)
        g_tangent = g_curr - proj[:, np.newaxis] * pts
        
        # Adaptive step size with line search
        alpha = lr
        for _ in range(10):
            new_pts = _retract(pts - alpha * g_tangent)
            f_new, g_new = _energy_and_grad(new_pts)
            if f_new < f_curr:
                pts = new_pts
                f_curr = f_new
                g_curr = g_new
                lr = alpha
                break
            alpha *= 0.5
        
        else:
            # If no improvement, reduce learning rate
            lr *= 0.5
            if lr < 1e-15:
                break
        
        # Check convergence
        g_norm = np.linalg.norm(g_tangent)
        if g_norm < 1e-14:
            break
    
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Generate multiple diverse initial configurations
    inits = []
    
    # 1. Golden-angle spiral (Fibonacci)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    inits.append(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))
    
    # 2. Random initialization with seed
    rng = np.random.default_rng(42)
    for seed in [0, 1, 2]:
        rng = np.random.default_rng(seed)
        pts_rand = rng.normal(size=(N, 3))
        pts_rand = _retract(pts_rand)
        inits.append(pts_rand)
    
    # 3. Grid-based initialization (better distribution)
    # Use a spherical grid with slight perturbation
    n_theta = 10
    n_phi = 10
    theta = np.linspace(0, np.pi, n_theta)
    phi_grid = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    pts_grid = []
    for t in theta:
        for p in phi_grid:
            pts_grid.append([np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)])
    pts_grid = np.array(pts_grid[:N])
    rng = np.random.default_rng(123)
    pts_grid += rng.normal(scale=0.05, size=pts_grid.shape)
    inits.append(_retract(pts_grid))
    
    best_pts = None
    best_energy = float('inf')
    
    # Optimize each initial configuration
    for idx, pts_init in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
        
        # Run L-BFGS-B
        pts_opt = _lbfgs_sphere(pts_init.flatten(), deadline - 5.0)
        
        # Polish with high-precision gradient descent
        pts_opt = _polish(pts_opt, deadline)
        
        f, _ = _energy_and_grad(pts_opt)
        if f < best_energy:
            best_energy = f
            best_pts = pts_opt
        
        # If we have time, try a second polish pass on the best result
        if time.time() < deadline - 10.0 and idx == 0:
            # Extra polish on the first (spiral) result which is usually best
            pts_extra = _polish(best_pts.copy(), deadline)
            f_extra, _ = _energy_and_grad(pts_extra)
            if f_extra < best_energy:
                best_energy = f_extra
                best_pts = pts_extra
    
    # Final validation and normalization
    if best_pts is None:
        best_pts = inits[0]
    
    best_pts = _retract(best_pts)
    
    # Ensure no points are too close
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 1e-9:
        # Fallback to spiral if points are too close
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        best_pts = _retract(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
