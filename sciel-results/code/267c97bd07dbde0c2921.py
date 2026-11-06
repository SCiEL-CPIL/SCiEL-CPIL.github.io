"""Optimized logarithmic energy minimizer for 150 points on the unit sphere."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150
DEADLINE_S = 140.0  # safety margin before 150s kill


def _log_energy(pts):
    """Compute sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    d = np.maximum(d, 1e-15)
    terms = -np.log(d)
    return math.fsum(terms.tolist())


def _energy_and_grad_flat(x_flat):
    """Compute energy and gradient for L-BFGS-B. x_flat is (N*3,) array."""
    n = N
    pts = x_flat.reshape(n, 3)
    # Normalize to ensure on sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    # Compute pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # n x n x 3
    dists_sq = np.sum(diff * diff, axis=2)  # n x n
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, 1.0)
    np.fill_diagonal(dists_sq, 1.0)
    
    # Energy: sum_{i<j} log(1/d) = -sum_{i<j} log(d)
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    d = np.maximum(d, 1e-15)
    energy = math.fsum((-np.log(d)).tolist())
    
    # Gradient: grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    inv_d2 = np.zeros_like(dists_sq)
    mask = dists_sq > 1e-24
    inv_d2[mask] = 1.0 / dists_sq[mask]
    np.fill_diagonal(inv_d2, 0.0)
    
    grad_3d = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)  # n x 3
    
    # Project to tangent space: grad_t = grad - (grad . x) * x
    dot = np.sum(grad_3d * pts, axis=1, keepdims=True)
    grad_t = grad_3d - dot * pts
    
    # Flatten gradient
    grad_flat = grad_t.flatten()
    
    return energy, grad_flat


def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using golden-angle spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    """Generate n random points on the unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _gradient_descent(pts, lr_init=0.01, max_iter=5000, tol=1e-12):
    """
    Gradient descent on the sphere for logarithmic energy minimization.
    Energy: E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    Gradient w.r.t. x_i: sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    Project gradient to tangent space: grad_t = grad - (grad . x_i) * x_i
    """
    n = len(pts)
    pts = pts.astype(np.float64).copy()
    lr = lr_init
    
    for it in range(max_iter):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        dists = np.sqrt(dists_sq)
        np.fill_diagonal(dists, 1.0)
        np.fill_diagonal(dists_sq, 1.0)
        
        inv_d2 = np.zeros_like(dists_sq)
        mask = dists_sq > 1e-24
        inv_d2[mask] = 1.0 / dists_sq[mask]
        np.fill_diagonal(inv_d2, 0.0)
        
        grad = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
        
        dot = np.sum(grad * pts, axis=1, keepdims=True)
        grad_t = grad - dot * pts
        
        pts_new = pts - lr * grad_t
        norms = np.linalg.norm(pts_new, axis=1, keepdims=True)
        pts_new = pts_new / norms
        
        step_norm = np.linalg.norm(pts_new - pts)
        if step_norm < tol:
            pts = pts_new
            break
        
        if it % 100 == 0 and it > 0:
            lr *= 0.95
        
        pts = pts_new
    
    return pts


def _lbfgs_optimize(pts, max_iter=2000, time_budget=30.0):
    """Use L-BFGS-B to optimize on the sphere with tangent space projection."""
    n = N
    x0 = pts.flatten()
    t_start = time.time()
    
    def objective(x_flat):
        if time.time() - t_start > time_budget:
            raise Exception("Time budget exceeded")
        energy, grad = _energy_and_grad_flat(x_flat)
        return energy, grad
    
    try:
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iter, 'gtol': 1e-10, 'ftol': 1e-12, 'maxfun': 5000}
        )
        pts_opt = result.x.reshape(n, 3)
        norms = np.linalg.norm(pts_opt, axis=1, keepdims=True)
        pts_opt = pts_opt / norms
        return pts_opt
    except Exception:
        return pts





def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)
    
    # Start with Fibonacci spiral
    best_pts = _fibonacci_sphere(N)
    best_energy = _log_energy(best_pts)
    
    def remaining():
        return DEADLINE_S - (time.time() - t0)
    
    # Phase 1: L-BFGS-B optimization from multiple starts
    # Phase 2: Gradient descent polishing
    
    # First, do L-BFGS-B from Fibonacci start
    if remaining() > 40.0:
        lbfgs_pts = _lbfgs_optimize(best_pts, max_iter=1500, time_budget=35.0)
        e = _log_energy(lbfgs_pts)
        if e < best_energy:
            best_energy = e
            best_pts = lbfgs_pts
    
    # Multiple restarts with L-BFGS-B
    n_restarts = 12
    
    for r in range(n_restarts):
        if remaining() < 25.0:
            break
        
        # Create perturbed starting point
        if r == 0:
            start = best_pts.copy()
        elif r < 4:
            # Small perturbations of best
            start = best_pts + 0.03 * rng.standard_normal((N, 3))
            start /= np.linalg.norm(start, axis=1, keepdims=True)
        elif r < 8:
            # Medium perturbations
            start = best_pts + 0.1 * rng.standard_normal((N, 3))
            start /= np.linalg.norm(start, axis=1, keepdims=True)
        else:
            # Random restarts
            start = _random_sphere(N, rng)
        
        # L-BFGS-B optimization
        time_budget = min(20.0, remaining() - 20.0)
        if time_budget > 5.0:
            opt_pts = _lbfgs_optimize(start, max_iter=800, time_budget=time_budget)
            e = _log_energy(opt_pts)
            if e < best_energy:
                best_energy = e
                best_pts = opt_pts
    
    # Final polishing with gradient descent
    if remaining() > 15.0:
        polished = _gradient_descent(best_pts, lr_init=0.002, max_iter=5000, tol=1e-14)
        e = _log_energy(polished)
        if e < best_energy:
            best_energy = e
            best_pts = polished
    
    # Another L-BFGS-B polish if time allows
    if remaining() > 10.0:
        final_lbfgs = _lbfgs_optimize(best_pts, max_iter=500, time_budget=min(8.0, remaining() - 5.0))
        e = _log_energy(final_lbfgs)
        if e < best_energy:
            best_energy = e
            best_pts = final_lbfgs
    
    # Final gradient descent polish
    if remaining() > 5.0:
        final_gd = _gradient_descent(best_pts, lr_init=0.0005, max_iter=3000, tol=1e-15)
        e = _log_energy(final_gd)
        if e < best_energy:
            best_energy = e
            best_pts = final_gd
    
    # Ensure valid output
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms
    
    # Validate
    assert best_pts.shape == (N, 3), f"Wrong shape: {best_pts.shape}"
    assert np.all(np.isfinite(best_pts)), "Non-finite values"
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    min_d = dists[iu].min()
    assert min_d > 1e-9, f"Points too close: {min_d}"
    
    return best_pts
