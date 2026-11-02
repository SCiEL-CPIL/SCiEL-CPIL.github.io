import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 150.0
DEADLINE_FRACTION = 0.85

def _log_energy_fast(pts):
    """Vectorized log energy: sum_{i<j} log(1/|xi-xj|)."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists2, 1.0)
    iu = np.triu_indices(n, k=1)
    # log(1/d) = -0.5 * log(d^2)
    return -0.5 * np.sum(np.log(dists2[iu]))

def _log_energy_grad_fast(pts):
    """Vectorized log energy and gradient (Euclidean gradient of the energy)."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists2, 1.0)
    
    iu = np.triu_indices(n, k=1)
    energy = -0.5 * np.sum(np.log(dists2[iu]))
    
    # E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # dE/dxi = -sum_{j!=i} (xi-xj)/|xi-xj|^2
    inv_dists2 = 1.0 / dists2
    np.fill_diagonal(inv_dists2, 0.0)
    
    s = (diff * inv_dists2[:, :, np.newaxis]).sum(axis=1)
    grad = -s
    
    return energy, grad

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _tangent_project(pts, grad):
    """Project gradient to tangent space of sphere."""
    coeffs = np.sum(grad * pts, axis=1, keepdims=True)
    return grad - coeffs * pts

def _riemannian_flow(pts, max_iter=2000, tol=1e-14):
    """Riemannian gradient flow with adaptive step size."""
    pts = _project_to_sphere(pts)
    current_energy = _log_energy_fast(pts)
    
    alpha = 0.05
    min_alpha = 1e-16
    max_alpha = 1.0
    stall_count = 0
    
    for _ in range(max_iter):
        e, g = _log_energy_grad_fast(pts)
        g_t = _tangent_project(pts, g)
        grad_norm = np.linalg.norm(g_t)
        
        if grad_norm < tol:
            break
        
        # Backtracking line search
        a = alpha
        found = False
        for _ in range(60):
            new_pts = pts - a * g_t
            new_pts = _project_to_sphere(new_pts)
            new_e = _log_energy_fast(new_pts)
            
            if new_e < current_energy - 1e-18 * a * grad_norm:
                pts = new_pts
                current_energy = new_e
                found = True
                break
            a *= 0.5
        
        if found:
            improvement = e - current_energy
            if improvement < 1e-16:
                stall_count += 1
            else:
                stall_count = 0
            alpha = min(a * 1.5, max_alpha)
        else:
            alpha = a
            if alpha < min_alpha:
                break
            if stall_count > 50:
                break
    
    return pts, current_energy

def _polish_high_precision(pts, deadline):
    """High-precision polishing with very tight tolerances."""
    pts = _project_to_sphere(pts)
    current_energy = _log_energy_fast(pts)
    
    alpha = 0.01
    min_alpha = 1e-18
    max_alpha = 0.1
    
    while time.time() < deadline:
        e, g = _log_energy_grad_fast(pts)
        g_t = _tangent_project(pts, g)
        grad_norm = np.linalg.norm(g_t)
        
        if grad_norm < 1e-18:
            break
        
        a = alpha
        found = False
        for _ in range(80):
            new_pts = pts - a * g_t
            new_pts = _project_to_sphere(new_pts)
            new_e = _log_energy_fast(new_pts)
            
            if new_e < current_energy - 1e-20 * a * grad_norm:
                pts = new_pts
                current_energy = new_e
                found = True
                break
            a *= 0.4
        
        if found:
            improvement = e - current_energy
            if improvement > 1e-18:
                alpha = min(a * 1.3, max_alpha)
            else:
                alpha = max(a * 0.8, min_alpha)
        else:
            alpha = max(a * 0.5, min_alpha)
            if alpha < min_alpha:
                break
    
    return pts, current_energy

def _fibonacci_spiral(n=N):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n=N, seed=None):
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    return _project_to_sphere(pts)

def _optimize_lbfgs(pts, deadline):
    """Run L-BFGS-B optimization with projected gradients."""
    pts = _project_to_sphere(pts)
    x0 = pts.flatten()
    
    def func(x):
        p = _project_to_sphere(x.reshape(-1, 3))
        e, _ = _log_energy_grad_fast(p)
        return e
        
    def jac(x):
        p = _project_to_sphere(x.reshape(-1, 3))
        _, g = _log_energy_grad_fast(p)
        g_t = _tangent_project(p, g)
        return g_t.flatten()
        
    options = {
        'maxiter': 10000,
        'gtol': 1e-15,
        'ftol': 1e-20,
        'disp': False,
        'maxfun': 50000,
        'maxls': 50,
    }
    
    try:
        result = minimize(func, x0, jac=jac, method='L-BFGS-B', options=options)
        if result.fun < 1e17 and time.time() < deadline:
            return _project_to_sphere(result.x.reshape(-1, 3))
    except Exception:
        pass
        
    return pts

def construct():
    global _start_time
    _start_time = time.time()
    deadline = _start_time + TIME_LIMIT * DEADLINE_FRACTION
    
    best_pts = None
    best_energy = float('inf')
    
    # Define diverse starting configurations
    rng = np.random.default_rng(12345)
    fib = _fibonacci_spiral(N)
    
    starts = [
        fib.copy(),
        _project_to_sphere(fib + rng.normal(0, 0.02, (N, 3))),
        _project_to_sphere(fib + rng.normal(0, 0.05, (N, 3))),
        _project_to_sphere(fib + rng.normal(0, 0.10, (N, 3))),
        _project_to_sphere(fib + rng.normal(0, 0.20, (N, 3))),
        _project_to_sphere(fib + rng.normal(0, 0.35, (N, 3))),
        _random_sphere(N, seed=1),
        _random_sphere(N, seed=7),
        _random_sphere(N, seed=13),
        _random_sphere(N, seed=42),
    ]
    
    # Deduplicate
    unique_starts = [starts[0]]
    for s in starts[1:]:
        if all(np.linalg.norm(s - u) > 0.2 for u in unique_starts):
            unique_starts.append(s)
    starts = unique_starts
    
    # Phase 1: L-BFGS optimization (60% of time)
    opt_deadline = _start_time + (deadline - _start_time) * 0.60
    
    for i, start in enumerate(starts):
        if time.time() > opt_deadline:
            break
            
        remaining = opt_deadline - time.time()
        if remaining < 2.0:
            break
        
        # Allocate time: first 3 get more, rest get less
        if i < 3:
            alloc = min(remaining, (deadline - _start_time) * 0.06)
        elif i < 6:
            alloc = min(remaining, (deadline - _start_time) * 0.04)
        else:
            alloc = min(remaining, (deadline - _start_time) * 0.025)
            
        sub_deadline = time.time() + alloc
        
        optimized = _optimize_lbfgs(start, sub_deadline)
        e = _log_energy_fast(optimized)
        
        if e < best_energy:
            best_energy = e
            best_pts = optimized.copy()
    
    # Phase 2: Riemannian flow polish (25% of time)
    if best_pts is not None:
        flow_deadline = _start_time + (deadline - _start_time) * 0.85
        best_pts, best_energy = _riemannian_flow(best_pts, max_iter=5000, tol=1e-15)
    
    # Phase 3: High-precision polish (remaining time with safety margin)
    if best_pts is not None:
        hp_deadline = deadline - 5.0
        best_pts, best_energy = _polish_high_precision(best_pts, hp_deadline)
        
        best_pts = _project_to_sphere(best_pts)
        
        # Final validity check
        if not np.all(np.isfinite(best_pts)):
            return _fibonacci_spiral(N)
            
        diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists)
        if min_d < 1e-9:
            return _fibonacci_spiral(N)
        
    else:
        best_pts = _fibonacci_spiral(N)
        
    return best_pts
