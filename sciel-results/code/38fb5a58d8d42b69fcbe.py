"""
Combined approach for logarithmic energy minimization of 150 points on the sphere.
Uses:
1. Diverse initialization (Fibonacci, Spectral, Sobol, Random)
2. Ambient-space L-BFGS optimization (inspired by the second solution)
3. Basin hopping for escaping local minima
4. High-precision Riemannian gradient descent polishing
"""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(N, k=1)
    return -math.fsum(np.log(dist[iu]).tolist())


def _energy_fast(pts):
    """Fast energy calculation using np.sum for optimization loops."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(N, k=1)
    return -np.sum(np.log(dist[iu]))


def _energy_and_gradient(pts):
    """
    Energy and gradient (in ambient R^3) of log energy for points ON the sphere.
    This is the tangent-space gradient.
    """
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient calculation:
    # grad_i = sum_{j!=i} - (x_i - x_j) / ||x_i - x_j||^2
    # Note: The term is -1/d * (x_i-x_j)/d = -(x_i-x_j)/d^2
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    # diff[i,j] = x_i - x_j
    # We want sum_j - diff[i,j] * inv_dist_sq[i,j]
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-14] = 1.0
    return pts / norms


def _fibonacci_points(n):
    """Fibonacci spiral (low-discrepancy) points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _sobol_points(n, seed=12345):
    """Quasi-random (Sobol) points on unit sphere."""
    from scipy.stats import qmc
    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    raw = sampler.random(n)
    pts = 2.0 * raw - 1.0
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-14] = 1.0
    return pts / norms


def _spectral_init(n):
    """Spectral initialization using repulsion."""
    np.random.seed(0)
    pts = np.random.randn(n, 3)
    pts = _project_to_sphere(pts)
    
    for it in range(50):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        dist = np.sqrt(dist_sq)
        
        force = 1.0 / (dist + 1e-10)
        np.fill_diagonal(force, 0.0)
        
        grad = np.sum(force[:, :, np.newaxis] * diff, axis=1)
        pts = pts + 0.01 * grad
        pts = _project_to_sphere(pts)
    
    return pts


def _compute_energy_and_grad_ambient(u):
    """
    Computes energy and gradient with respect to ambient coordinates u.
    u is shape (N, 3). Internally normalizes to x.
    This allows L-BFGS to optimize in R^3 directly.
    """
    u_norms = np.linalg.norm(u, axis=1, keepdims=True)
    # Avoid division by zero
    u_norms = np.where(u_norms < 1e-12, 1e-12, u_norms)
    x = u / u_norms
    
    # Compute pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Energy
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient with respect to x (tangent space gradient on sphere)
    # grad_x_i = sum_{j!=i} - (x_i - x_j) / ||x_i - x_j||^2
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad_x = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    # Gradient with respect to u
    # grad_u_i = (1/||u_i||) * (grad_x_i - (grad_x_i . x_i) x_i)
    # Since grad_x_i is orthogonal to x_i (it's a tangent vector), the dot product is 0.
    # So grad_u_i = grad_x_i / ||u_i||
    # However, due to numerical errors, it's safer to project.
    # Let's check orthogonality.
    # grad_x_i . x_i = sum_j - (x_i - x_j) . x_i / d^2 = sum_j - (1 - x_i.x_j)/d^2
    # This is not strictly 0 in finite precision?
    # Actually, the gradient of a function on the sphere is always tangent.
    # So grad_x_i . x_i should be 0.
    # We can just divide by norm.
    u_norms_flat = u_norms.flatten()
    grad_u = grad_x / u_norms_flat[:, np.newaxis]
    
    return energy, grad_u.flatten()


def _optimize_on_sphere(pts, deadline, maxiter=2000, ftol=1e-14, gtol=1e-12):
    """
    Optimize log energy on the sphere using L-BFGS-B in ambient space.
    """
    pts = _project_to_sphere(pts.copy())
    x0 = pts.flatten()
    
    best = pts.copy()
    best_e = _log_energy(pts)
    
    try:
        res = minimize(
            _compute_energy_and_grad_ambient,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol,
                     'maxls': 40, 'maxcor': 30}
        )
        
        if time.time() < deadline:
            p1 = _project_to_sphere(res.x.reshape(N, 3))
            e1 = _log_energy(p1)
            if e1 < best_e:
                best, best_e = p1, e1
                
    except Exception:
        pass
        
    return best, best_e


def _tangent_grad_descent_polish(pts, deadline, max_iter=10000, base_lr=1e-3):
    """High-precision line-search gradient descent on the sphere (Riemannian)."""
    pts = _project_to_sphere(pts.copy())
    best = pts.copy()
    best_e = _log_energy(pts)
    current_e = best_e
    lr = base_lr
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        e, g = _energy_and_gradient(pts)
        # Project gradient to tangent space
        g_tan = g - np.sum(g * pts, axis=1, keepdims=True) * pts
        gnorm = np.linalg.norm(g_tan)
        if gnorm < 1e-14:
            break
        
        # Line search (backtracking Armijo)
        alpha = lr
        improved = False
        for _ in range(40):
            cand = pts - alpha * g_tan
            cand = _project_to_sphere(cand)
            ce = _energy_fast(cand)
            if ce < current_e - 1e-4 * alpha * gnorm * gnorm:
                pts = cand
                current_e = ce
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            lr *= 0.5
            if lr < 1e-10:
                break
            continue
        
        if current_e < best_e - 1e-15:
            best = pts.copy()
            best_e = current_e
        
        # Occasionally increase learning rate if stuck
        if it % 100 == 0 and lr < base_lr:
            lr = base_lr * 0.1
    
    return best, best_e


def _basin_hopping(pts, deadline, max_iters=20, noise_scales=None):
    """Basin hopping: perturb and re-optimize to escape local minima."""
    if noise_scales is None:
        noise_scales = [0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.06, 0.1, 0.15, 0.2]
    
    current_pts = pts.copy()
    current_energy = _log_energy(current_pts)
    best_pts = current_pts.copy()
    best_energy = current_energy
    
    np.random.seed(42)  # Reproducibility
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
        
        # Choose noise scale
        scale = noise_scales[i % len(noise_scales)]
        noise = np.random.normal(0, scale, current_pts.shape)
        perturbed = _project_to_sphere(current_pts + noise)
        
        # Optimize the perturbed point with time budget
        alloc = max(3.0, min(15.0, deadline - time.time() - 5.0))
        if alloc < 2.0:
            break
        opt_pts, opt_e = _optimize_on_sphere(perturbed, time.time() + alloc, maxiter=500)
        
        if opt_e < current_energy:
            current_pts = opt_pts
            current_energy = opt_e
            if current_energy < best_energy:
                best_pts = current_pts.copy()
                best_energy = current_energy
    
    return best_pts, best_energy


def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Multiple diverse starting configurations
    start_configs = [
        _fibonacci_points(N),
        _spectral_init(N),
        _sobol_points(N, seed=12345),
        _sobol_points(N, seed=99999),
    ]
    
    # Add some random starts for diversity
    rng = np.random.default_rng(42)
    for _ in range(2):
        u = rng.normal(size=(N, 3))
        start_configs.append(_project_to_sphere(u))

    # Optimize each starting config with time budget
    time_per_config = 15.0  # Reduced slightly to allow more configs if needed
    for i, cfg in enumerate(start_configs):
        remaining = deadline - time.time()
        if remaining < 10.0:
            break
        alloc = min(time_per_config, remaining - 20.0)
        if alloc < 3.0:
            break
        opt_pts, opt_e = _optimize_on_sphere(cfg, time.time() + alloc, maxiter=1000)
        if opt_e < best_energy:
            best_energy = opt_e
            best_pts = opt_pts.copy()
    
    if best_pts is None:
        best_pts = _fibonacci_points(N)
        best_energy = _log_energy(best_pts)
    
    # Phase 2: Basin hopping to escape local minima
    if time.time() < deadline - 30.0:
        bh_deadline = deadline - 15.0
        current_pts = best_pts.copy()
        current_energy = best_energy
        
        noise_scales = [0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.06, 0.1, 0.15, 0.2,
                        0.04, 0.07, 0.11, 0.16, 0.22, 0.05, 0.09, 0.14, 0.19, 0.24]
        
        for i in range(len(noise_scales)):
            if time.time() > bh_deadline:
                break
            
            scale = noise_scales[i]
            np.random.seed(1000 + i)
            noise = np.random.normal(0, scale, current_pts.shape)
            perturbed = _project_to_sphere(current_pts + noise)
            
            alloc = min(12.0, bh_deadline - time.time() - 5.0)
            if alloc < 3.0:
                break
            opt_pts, opt_e = _optimize_on_sphere(perturbed, time.time() + alloc, maxiter=600)
            
            if opt_e < current_energy:
                current_pts = opt_pts
                current_energy = opt_e
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pts = opt_pts.copy()
    
    # Phase 3: Polish with gradient descent
    if time.time() < deadline - 10.0:
        p, e = _tangent_grad_descent_polish(best_pts, deadline - 5.0, max_iter=15000, base_lr=1e-3)
        if e < best_energy:
            best_energy = e
            best_pts = p
    
    # Final L-BFGS tight pass
    if time.time() < deadline - 5.0:
        p, e = _optimize_on_sphere(best_pts, deadline - 2.0, maxiter=3000, ftol=1e-16, gtol=1e-14)
        if e < best_energy:
            best_energy = e
            best_pts = p
    
    best_pts = _project_to_sphere(best_pts)
    
    # Validate: no points closer than 1e-9, finite values
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        best_pts = _project_to_sphere(_fibonacci_points(N))
    d = np.linalg.norm(best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(d, 1.0)
    if np.min(d) < 1e-9:
        best_pts = _project_to_sphere(_fibonacci_points(N))
    
    return best_pts
# EVOLVE-BLOCK-END
