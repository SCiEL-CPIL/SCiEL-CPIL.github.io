"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
TIME_LIMIT = 135.0  # Safety margin before 150s kill

def _log_energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|xi-xj|)
    and the Riemannian gradient (projected to tangent space) on the sphere.
    
    Euclidean gradient for point i: sum_{j!=i} (x_j - x_i) / |x_i - x_j|^2
    Riemannian gradient = Euclidean grad - (Euclidean grad . x_i) * x_i
    """
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum over i<j of -log(dist_ij)
    iu = np.triu_indices(N, k=1)
    e = -np.sum(np.log(dist[iu]))
    
    # Euclidean gradient: grad_i = sum_{j!=i} (x_j - x_i) / d_ij^2
    # = -sum_{j!=i} (x_i - x_j) / d_ij^2
    inv_d2 = 1.0 / dist_sq  # (N, N)
    np.fill_diagonal(inv_d2, 0.0)
    
    # diff[i,j] = x_i - x_j, so we need (x_j - x_i) = -diff[i,j]
    # grad_i = sum_j (-diff[i,j]) * inv_d2[i,j] = -sum_j diff[i,j] * inv_d2[i,j]
    euc_grad = -np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    
    # Project to tangent space: Riemannian grad = euc_grad - (euc_grad . x_i) x_i
    dot_products = np.sum(euc_grad * pts, axis=1)  # (N,)
    rgrad = euc_grad - dot_products[:, np.newaxis] * pts
    
    return e, rgrad


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _fibonacci_spiral(n):
    """Generate n points on sphere using golden angle spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _retract(pts, grad, step_size):
    """
    Retraction on the sphere: move along the Riemannian gradient direction,
    then project back to the sphere.
    grad is the Riemannian gradient (tangent vectors).
    """
    new_pts = pts + step_size * grad
    return _project_to_sphere(new_pts)


def _riemannian_gradient_descent(pts, time_budget, lr_init=0.01, lr_decay=0.9999):
    """
    Riemannian gradient descent with adaptive step size.
    """
    t_start = time.time()
    n = len(pts)
    current_energy, rgrad = _log_energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = current_energy
    
    lr = lr_init
    patience = 0
    max_patience = 200
    
    while time.time() - t_start < time_budget:
        if time.time() - t_start >= time_budget:
            break
        
        e, rgrad = _log_energy_and_grad(pts)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
            patience = 0
        else:
            patience += 1
        
        if patience > max_patience:
            break
        
        # Check convergence
        gnorm = np.linalg.norm(rgrad)
        if gnorm < 1e-12:
            break
        
        # Adaptive step size: normalize gradient, use fixed step
        # Step size proportional to 1/sqrt(gnorm) for stability
        step = lr / (1.0 + gnorm)
        
        new_pts = _retract(pts, -rgrad, step)
        
        # Check if it improved
        new_e, _ = _log_energy_and_grad(new_pts)
        if new_e < e:
            pts = new_pts
            current_energy = new_e
            lr = min(lr * 1.05, 0.1)
        else:
            lr *= 0.5
            if lr < 1e-8:
                break
    
    return best_pts, best_energy


def _lbfgs_polish(pts, time_budget):
    """
    L-BFGS optimization using scipy, operating on the flattened sphere coordinates
    with a projection-based approach.
    """
    t_start = time.time()
    n = len(pts)
    
    def objective_and_grad(x_flat):
        if time.time() - t_start > time_budget:
            return np.inf, np.zeros_like(x_flat)
        x = x_flat.reshape(n, 3)
        x = _project_to_sphere(x)
        e, rgrad = _log_energy_and_grad(x)
        return e, rgrad.flatten()
    
    x0 = pts.flatten()
    
    # Use L-BFGS-B with bounds to keep components in [-1, 1]
    # This isn't perfect for sphere constraint but helps
    result = minimize(
        objective_and_grad,
        x0,
        jac=True,
        method='L-BFGS-B',
        options={
            'maxiter': 2000,
            'maxfun': 5000,
            'ftol': 1e-15,
            'gtol': 1e-10,
            'maxcor': 20
        }
    )
    
    if result.success or result.fun < _log_energy_and_grad(pts)[0]:
        best_pts = _project_to_sphere(result.x.reshape(n, 3))
        best_e = _log_energy_and_grad(best_pts)[0]
        return best_pts, best_e
    
    return pts, _log_energy_and_grad(pts)[0]


def _basin_hopping(pts, time_budget, n_restarts=5):
    """
    Simple basin hopping: perturb points, then polish with gradient descent.
    """
    t_start = time.time()
    best_pts = pts.copy()
    best_energy = _log_energy_and_grad(pts)[0]
    rng = np.random.default_rng(123)
    
    remaining = time_budget - (time.time() - t_start)
    if remaining < 5:
        return best_pts, best_energy
    
    per_restart = remaining / n_restarts
    
    for restart in range(n_restarts):
        if time.time() - t_start >= time_budget:
            break
        
        # Create perturbed version
        perturbed = pts.copy()
        noise_scale = 0.1 * (1.0 - restart / n_restarts)
        for _ in range(3):
            noise = rng.standard_normal((len(pts), 3)) * noise_scale
            perturbed = _project_to_sphere(perturbed + noise)
        
        # Polish with gradient descent
        gd_budget = per_restart * 0.7
        polished, e_polished = _riemannian_gradient_descent(perturbed, gd_budget, lr_init=0.005)
        
        if e_polished < best_energy:
            best_energy = e_polished
            best_pts = polished.copy()
    
    return best_pts, best_energy


def _final_polish(pts, time_budget):
    """
    Final high-precision polish using L-BFGS with tight tolerances.
    """
    t_start = time.time()
    n = len(pts)
    
    def objective_and_grad(x_flat):
        if time.time() - t_start > time_budget:
            return np.inf, np.zeros_like(x_flat)
        x = x_flat.reshape(n, 3)
        x = _project_to_sphere(x)
        e, rgrad = _log_energy_and_grad(x)
        return e, rgrad.flatten()
    
    x0 = pts.flatten()
    
    result = minimize(
        objective_and_grad,
        x0,
        jac=True,
        method='L-BFGS-B',
        options={
            'maxiter': 5000,
            'maxfun': 10000,
            'ftol': 1e-18,
            'gtol': 1e-12,
            'maxcor': 30
        }
    )
    
    if result.fun < _log_energy_and_grad(pts)[0]:
        best_pts = _project_to_sphere(result.x.reshape(n, 3))
        best_e = _log_energy_and_grad(best_pts)[0]
        return best_pts, best_e
    
    return pts, _log_energy_and_grad(pts)[0]


def construct():
    t_start = time.time()
    
    # Start with Fibonacci spiral
    pts = _fibonacci_spiral(N)
    
    # Phase 1: Riemannian gradient descent (fast, gets us close)
    remaining = TIME_LIMIT - (time.time() - t_start)
    gd_budget = min(20.0, remaining * 0.2)
    pts, e_gd = _riemannian_gradient_descent(pts, gd_budget, lr_init=0.02)
    
    # Phase 2: Basin hopping to escape local minima
    remaining = TIME_LIMIT - (time.time() - t_start)
    bh_budget = min(40.0, remaining * 0.35)
    pts, e_bh = _basin_hopping(pts, bh_budget, n_restarts=6)
    
    # Phase 3: L-BFGS polish
    remaining = TIME_LIMIT - (time.time() - t_start)
    lbfgs_budget = min(30.0, remaining * 0.3)
    pts, e_lbfgs = _lbfgs_polish(pts, lbfgs_budget)
    
    # Phase 4: Final high-precision polish
    remaining = TIME_LIMIT - (time.time() - t_start)
    final_budget = min(remaining * 0.6, 20.0)
    pts, e_final = _final_polish(pts, final_budget)
    
    # Final check and validation
    pts = _project_to_sphere(pts)
    
    if not np.all(np.isfinite(pts)):
        pts = _fibonacci_spiral(N)
    
    # Ensure minimum separation
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                # Nudge point j slightly
                pts[j] = pts[j] + 1e-8 * np.random.randn(3)
                pts[j] = pts[j] / np.linalg.norm(pts[j])
    
    return pts

# EVOLVE-BLOCK-END
