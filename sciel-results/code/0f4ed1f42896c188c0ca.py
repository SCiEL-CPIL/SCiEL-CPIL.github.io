"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

_start_time = time.time()
_TIME_LIMIT = 110.0  # Safety margin before 120s kill

def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient w.r.t. x_i: sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    pts = np.asarray(pts)
    n = len(pts)
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dist_sq, 1.0)  # Avoid division by zero
    dist = np.sqrt(dist_sq)
    
    # Energy: -sum_{i<j} log(d_ij)
    log_dist = np.log(dist)
    upper_tri = np.triu_indices(n, k=1)
    energy = -np.sum(log_dist[upper_tri])
    
    # Gradient: for each i, grad_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # = sum_{j!=i} diff[i,j] / dist_sq[i,j]
    with np.errstate(divide='ignore', invalid='ignore'):
        weight = np.zeros_like(dist_sq)
        weight[upper_tri] = 1.0 / dist_sq[upper_tri]
        weight = weight + weight.T  # Symmetrize
    grad = np.sum(weight[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad

def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _riemannian_descent(pts, max_iter=5000, lr_init=0.01, lr_min=1e-6):
    """Riemannian gradient descent with backtracking line search on the sphere."""
    pts = _project_to_sphere(pts.copy())
    n = len(pts)
    best_energy, _ = _energy_and_grad(pts)
    best_pts = pts.copy()
    lr = lr_init
    
    for it in range(max_iter):
        if time.time() - _start_time > _TIME_LIMIT - 5:
            break
        
        energy, grad = _energy_and_grad(pts)
        
        # Riemannian gradient: project Euclidean gradient tangent to sphere
        # Tangent space at x: {v : v·x = 0}
        # Project: grad_tangent = grad - (grad·x) x
        grad_dot_x = np.sum(grad * pts, axis=1, keepdims=True)
        grad_tangent = grad - grad_dot_x * pts
        
        # Backtracking line search
        step = lr
        for _ in range(20):
            # Update: move in tangent direction, then project back to sphere
            new_pts = pts + step * grad_tangent
            new_pts = _project_to_sphere(new_pts)
            
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < energy:
                break
            step *= 0.5
        
        pts = new_pts
        if new_energy < best_energy:
            best_energy = new_energy
            best_pts = pts.copy()
        
        # Adaptive learning rate
        if new_energy < energy - 1e-10:
            lr = min(lr * 1.05, 0.1)
        else:
            lr = max(lr * 0.8, lr_min)
        
        # Convergence check
        if np.linalg.norm(grad_tangent, ord=np.inf) < 1e-10:
            break
    
    return best_pts, best_energy

def _lbfgs_optimize(pts):
    """Polish with L-BFGS-B in ambient space with sphere constraint."""
    if time.time() - _start_time > _TIME_LIMIT - 10:
        return pts, _energy_and_grad(pts)[0]
    
    def objective_and_grad(x):
        pts = x.reshape(-1, 3)
        e, g = _energy_and_grad(pts)
        return e, g.ravel()
    
    res = minimize(objective_and_grad, pts.ravel(), method='L-BFGS-B',
                   jac=True, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
    
    # Project result to sphere
    pts_res = _project_to_sphere(res.x.reshape(-1, 3))
    return pts_res, _energy_and_grad(pts_res)[0]

def construct():
    global _start_time
    _start_time = time.time()
    
    # Phase 1: Fibonacci spiral initialization
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts, best_energy = pts, _energy_and_grad(pts)[0]
    
    # Phase 2: Riemannian gradient descent
    if time.time() - _start_time < _TIME_LIMIT - 30:
        pts_rd, energy_rd = _riemannian_descent(pts, max_iter=3000)
        if energy_rd < best_energy:
            best_pts, best_energy = pts_rd, energy_rd
    
    # Phase 3: L-BFGS-B polish
    if time.time() - _start_time < _TIME_LIMIT - 15:
        pts_lbfgs, energy_lbfgs = _lbfgs_optimize(best_pts)
        if energy_lbfgs < best_energy:
            best_pts, best_energy = pts_lbfgs, energy_lbfgs
    
    # Phase 4: Additional random restarts if time permits
    rng = np.random.default_rng(42)
    remaining = _TIME_LIMIT - (time.time() - _start_time)
    if remaining > 20:
        n_restarts = min(5, int(remaining // 15))
        for restart in range(n_restarts):
            if time.time() - _start_time > _TIME_LIMIT - 10:
                break
            
            # Random initialization
            pts_rand = rng.normal(size=(N, 3))
            pts_rand = _project_to_sphere(pts_rand)
            
            # Quick Riemannian descent
            if time.time() - _start_time < _TIME_LIMIT - 15:
                pts_r, energy_r = _riemannian_descent(pts_rand, max_iter=1000, lr_init=0.005)
                if energy_r < best_energy:
                    best_pts, best_energy = pts_r, energy_r
                
                # Polish
                if time.time() - _start_time < _TIME_LIMIT - 5:
                    pts_p, energy_p = _lbfgs_optimize(best_pts)
                    if energy_p < best_energy:
                        best_pts, best_energy = pts_p, energy_p
    
    # Final validation
    best_pts = _project_to_sphere(best_pts)
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-9) or not np.all(np.isfinite(best_pts)):
        # Fallback to Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    return best_pts
# EVOLVE-BLOCK-END
