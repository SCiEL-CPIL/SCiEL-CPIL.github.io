import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def _fibonacci_sphere(n):
    """Generate points using Fibonacci spiral on sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts

def _halton_sphere(n, base):
    """Generate points using Halton sequence on sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    pts = []
    for i in range(1, n + 1):
        u = halton(i, base)
        v = halton(i, base + 1)
        theta = 2.0 * np.pi * u
        phi = np.arccos(1.0 - 2.0 * v)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        pts.append([x, y, z])
    return np.array(pts)

def _log_energy_fast(pts):
    """Vectorized log energy computation using log(1/d) = -log(d)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    # Use log(1/d) = -0.5 * log(d^2) for numerical stability
    dist_sq = np.maximum(dist_sq, 1e-15)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    energy = -0.5 * np.sum(np.log(dist_sq[mask]))
    return energy

def _log_energy_grad(pts):
    """Compute energy and gradient of log energy on sphere."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    dist_sq = np.maximum(dist_sq, 1e-15)
    
    # Energy: sum_{i<j} log(1/d_ij) = -0.5 * sum_{i<j} log(d_ij^2)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    energy = -0.5 * np.sum(np.log(dist_sq[mask]))
    
    # Gradient: d/dx_i sum_j log(1/|x_i - x_j|) = sum_j (x_i - x_j) / |x_i - x_j|^2
    # Project to tangent space
    forces = np.sum(diff / dist_sq[:, :, np.newaxis], axis=1)
    dot_products = np.sum(forces * pts, axis=1)
    proj_forces = forces - dot_products[:, np.newaxis] * pts
    return energy, proj_forces

def _exponential_map(pts, tangent):
    """Apply exponential map on sphere: move along geodesic."""
    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    angles = norms
    sin_angles = np.sin(angles)
    cos_angles = np.cos(angles)
    # p_new = cos(|v|)*p + sin(|v|)/|v| * v
    new_pts = cos_angles * pts + (sin_angles / norms) * tangent
    # Normalize to unit sphere
    new_norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    new_norms = np.maximum(new_norms, 1e-15)
    new_pts = new_pts / new_norms
    return new_pts

def _riemannian_cg(pts, max_iter=200, tol=1e-10):
    """Riemannian conjugate gradient method on the sphere."""
    best_pts = pts.copy()
    best_energy = _log_energy_fast(pts)
    
    for k in range(max_iter):
        energy, grad = _log_energy_grad(pts)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < tol:
            break
        
        # Search direction
        if k == 0:
            d = -grad
        else:
            # Polak-Ribiere beta
            grad_prev = prev_grad
            beta = np.sum(d * (grad - grad_prev)) / np.sum(grad_prev * grad_prev)
            beta = max(beta, 0.0)
            d = -grad + beta * d
        
        # Line search: backtracking
        step = 0.1
        for ls_iter in range(30):
            new_pts = _exponential_map(pts, step * d)
            new_energy = _log_energy_fast(new_pts)
            
            if new_energy < energy - 1e-4 * step * np.sum(grad * d):
                break
            step *= 0.5
        
        pts = new_pts
        prev_grad = grad
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pts = new_pts.copy()
        
        if step < 1e-8:
            break
    
    return best_pts, best_energy

def _lbfgs_optimize(pts, max_iter=500, ftol=1e-12, gtol=1e-8):
    """L-BFGS-B optimization in 3D space with normalization."""
    x0 = pts.flatten()
    
    def objective(x):
        p = x.reshape(N, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        p = p / norms
        return _log_energy_fast(p)
    
    def gradient_func(x):
        p = x.reshape(N, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        p = p / norms
        _, grad = _log_energy_grad(p)
        return grad.flatten()
    
    try:
        result = minimize(
            objective, x0, jac=gradient_func,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50}
        )
        if result.success or result.fun < _log_energy_fast(pts):
            res_pts = result.x.reshape(N, 3)
            norms = np.linalg.norm(res_pts, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            res_pts = res_pts / norms
            return res_pts, result.fun
    except Exception:
        pass
    
    return pts, _log_energy_fast(pts)

def _polish(pts, max_iter=1000, tol=1e-12):
    """High-precision polish with very tight tolerances."""
    # First, do a CG pass
    pts, energy = _riemannian_cg(pts, max_iter=max_iter, tol=tol)
    
    # Then L-BFGS with tight tolerances
    pts, energy = _lbfgs_optimize(pts, max_iter=500, ftol=tol, gtol=tol)
    
    # Final CG pass
    pts, energy = _riemannian_cg(pts, max_iter=max_iter, tol=tol)
    
    return pts, energy

def _calc_energy_fsum(pts):
    """High-precision energy using math.fsum, matching evaluator."""
    terms = []
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit
    
    # Initial configuration: Fibonacci spiral
    pts = _fibonacci_sphere(N)
    best_pts = pts.copy()
    best_energy = _log_energy_fast(pts)
    
    # Phase 1: Riemannian CG from Fibonacci
    if time.time() < deadline - 40.0:
        pts, energy = _riemannian_cg(pts, max_iter=500, tol=1e-8)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Phase 2: L-BFGS polish
    if time.time() < deadline - 30.0:
        pts, energy = _lbfgs_optimize(pts, max_iter=500, ftol=1e-10, gtol=1e-8)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Phase 3: Try Halton initialization and optimize
    if time.time() < deadline - 20.0:
        pts_halton = _halton_sphere(N, 2)
        pts_halton, energy_halton = _riemannian_cg(pts_halton, max_iter=300, tol=1e-8)
        if energy_halton < best_energy:
            best_energy = energy_halton
            best_pts = pts_halton.copy()
    
    # Phase 4: High-precision polish
    if time.time() < deadline - 10.0:
        pts, energy = _polish(best_pts, max_iter=500, tol=1e-12)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Final verification with fsum
    final_energy = _calc_energy_fsum(best_pts)
    if math.isfinite(final_energy) and final_energy < best_energy:
        best_energy = final_energy
    
    # Ensure all points are on unit sphere
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms
    
    return best_pts
