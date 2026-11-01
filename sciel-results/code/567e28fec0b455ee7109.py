"""Dual-representation optimization for logarithmic energy minimization on sphere."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time


def _energy_and_grad(pts):
    """Compute logarithmic energy and its gradient for points on the unit sphere.
    
    Energy: sum_{i<j} log(1 / ||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)
    Gradient w.r.t. x_i: sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dist_sq[iu])
    energy = -math.fsum(math.log(d) for d in dists)
    
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _halton_init(n):
    """Generate n points using Halton sequence mapped to sphere."""
    def halton_sequence(n, base):
        seq = np.zeros(n)
        for i in range(1, n + 1):
            result = 0.0
            f = 1.0 / base
            num = i
            while num > 0:
                num, rem = divmod(num, base)
                result += rem * f
                f /= base
            seq[i - 1] = result
        return seq
    
    bases = [2, 3, 5]
    u = halton_sequence(n, bases[0])
    v = halton_sequence(n, bases[1])
    
    theta = 2.0 * np.pi * u
    phi = np.arccos(1.0 - 2.0 * v)
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    pts = np.column_stack([x, y, z])
    return pts


def _riemannian_cg(pts, max_iter=300, tol=1e-12):
    """Riemannian conjugate gradient with backtracking line search."""
    x = pts.copy()
    energy, grad_e = _energy_and_grad(x)
    
    inner = np.sum(grad_e * x, axis=1)
    grad_r = grad_e - inner[:, np.newaxis] * x
    
    p = -grad_r.copy()
    grad_r_prev = grad_r.copy()
    
    for it in range(max_iter):
        grad_norm = np.linalg.norm(grad_r)
        if grad_norm < tol:
            break
        
        p_norm = np.linalg.norm(p, axis=1, keepdims=True)
        p_norm[p_norm < 1e-12] = 1e-12
        p_unit = p / p_norm
        
        # Backtracking line search with Armijo condition
        alpha = 0.1
        c = 1e-4
        best_t = 0.0
        best_energy = energy
        
        for _ in range(20):
            x_trial = x * np.cos(alpha * p_norm) + p_unit * np.sin(alpha * p_norm)
            x_trial = _project_to_sphere(x_trial)
            e_trial, _ = _energy_and_grad(x_trial)
            
            if e_trial < energy - c * alpha * np.sum(grad_r * p):
                best_t = alpha
                best_energy = e_trial
                break
            alpha *= 0.5
        
        if best_t == 0.0:
            break
        
        x = x * np.cos(best_t * p_norm) + p_unit * np.sin(best_t * p_norm)
        x = _project_to_sphere(x)
        energy = best_energy
        
        _, grad_e_new = _energy_and_grad(x)
        inner_new = np.sum(grad_e_new * x, axis=1)
        grad_r_new = grad_e_new - inner_new[:, np.newaxis] * x
        
        # Polak-Ribiere beta
        beta_num = np.sum(grad_r_new * (grad_r_new - grad_r_prev))
        beta_den = np.sum(grad_r_prev * grad_r_prev)
        beta = max(0.0, beta_num / beta_den) if beta_den > 1e-14 else 0.0
        
        p = -grad_r_new + beta * p
        inner_p = np.sum(p * x, axis=1)
        p = p - inner_p[:, np.newaxis] * x
        
        grad_r_prev = grad_r_new
    
    return x, energy


def _projected_gradient(pts, max_iter=200, lr_init=0.02):
    """Projected gradient with adaptive step size."""
    x = pts.copy()
    lr = lr_init
    
    for it in range(max_iter):
        energy, grad_e = _energy_and_grad(x)
        
        inner = np.sum(grad_e * x, axis=1)
        grad_t = grad_e - inner[:, np.newaxis] * x
        
        grad_norm = np.linalg.norm(grad_t)
        if grad_norm < 1e-14:
            break
        
        improved = False
        for scale in [1.0, 0.5, 0.25, 0.1, 0.05]:
            t = lr * scale
            x_trial = x - t * grad_t
            x_trial = _project_to_sphere(x_trial)
            e_trial, _ = _energy_and_grad(x_trial)
            if e_trial < energy:
                x = x_trial
                energy = e_trial
                improved = True
                break
        
        if not improved:
            lr *= 0.5
            if lr < 1e-12:
                break
        else:
            lr = min(lr * 1.2, 0.1)
    
    return x, energy


def construct():
    start_time = time.time()
    time_budget = 115.0  # Leave 5s buffer
    
    # Initialize with Halton sequence
    pts = _halton_init(N)
    pts = _project_to_sphere(pts)
    
    # Phase 1: Coarse optimization with projected gradient
    pts, _ = _projected_gradient(pts, max_iter=100, lr_init=0.02)
    
    # Phase 2: Riemannian CG for better convergence
    pts, _ = _riemannian_cg(pts, max_iter=200, tol=1e-11)
    
    # Check time
    if time.time() - start_time > time_budget * 0.6:
        pts = _project_to_sphere(pts)
        return pts
    
    # Phase 3: Fine-tune with projected gradient
    pts, _ = _projected_gradient(pts, max_iter=100, lr_init=0.01)
    
    # Check time
    if time.time() - start_time > time_budget * 0.8:
        pts = _project_to_sphere(pts)
        return pts
    
    # Phase 4: Final Riemannian CG polish
    pts, _ = _riemannian_cg(pts, max_iter=150, tol=1e-13)
    
    # Ensure validity
    pts = _project_to_sphere(pts)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                pts[j] += 1e-8 * np.random.randn(3)
                pts[j] = _project_to_sphere(pts[j])
    
    if not np.all(np.isfinite(pts)):
        pts = _halton_init(N)
        pts = _project_to_sphere(pts)
    
    return pts
# EVOLVE-BLOCK-END
