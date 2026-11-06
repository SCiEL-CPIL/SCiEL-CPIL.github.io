"""Optimized logarithmic energy minimizer for 150 points on the unit sphere."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150
DEADLINE_S = 138.0  # safety margin before 150s kill


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
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    
    # Compute pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # n x n x 3
    dists_sq = np.sum(diff * diff, axis=2)  # n x n
    np.fill_diagonal(dists_sq, 1.0)
    
    # Energy: sum_{i<j} log(1/d) = -sum_{i<j} log(d)
    iu = np.triu_indices(n, k=1)
    d_sq = dists_sq[iu]
    d = np.sqrt(np.maximum(d_sq, 1e-30))
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


def _gradient_descent(pts, lr_init=0.01, max_iter=5000, tol=1e-14):
    """
    Gradient descent on the sphere for logarithmic energy minimization with adaptive step size.
    Uses backtracking line search for robustness.
    """
    n = len(pts)
    pts = pts.astype(np.float64).copy()
    lr = lr_init
    best_e = _log_energy(pts)
    
    for it in range(max_iter):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dists_sq, 1.0)
        
        inv_d2 = np.zeros_like(dists_sq)
        mask = dists_sq > 1e-24
        inv_d2[mask] = 1.0 / dists_sq[mask]
        np.fill_diagonal(inv_d2, 0.0)
        
        grad = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
        
        dot = np.sum(grad * pts, axis=1, keepdims=True)
        grad_t = grad - dot * pts
        
        # Adaptive step size with backtracking
        step_size = lr
        accepted = False
        for _ in range(10):
            pts_new = pts - step_size * grad_t
            norms = np.linalg.norm(pts_new, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            pts_new = pts_new / norms
            
            e_new = _log_energy(pts_new)
            if e_new < best_e:
                pts = pts_new
                best_e = e_new
                accepted = True
                break
            step_size *= 0.5
        
        if not accepted:
            # If no step accepted, try smaller initial lr
            lr *= 0.5
            if lr < 1e-10:
                break
        
        # Decay learning rate slowly
        if it % 50 == 0 and it > 0:
            lr *= 0.98
        
        step_norm = np.linalg.norm(pts - (pts + step_size * grad_t))
        if step_norm < tol:
            break
    
    return pts


def _lbfgs_optimize(pts, max_iter=2000, time_budget=30.0, gtol=1e-12):
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
            options={'maxiter': max_iter, 'gtol': gtol, 'ftol': 1e-14, 'maxfun': max_iter * 3}
        )
        pts_opt = result.x.reshape(n, 3)
        norms = np.linalg.norm(pts_opt, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
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
    
    # Phase 1: L-BFGS-B optimization from Fibonacci start
    if remaining() > 35.0:
        lbfgs_pts = _lbfgs_optimize(best_pts, max_iter=2000, time_budget=30.0, gtol=1e-12)
        e = _log_energy(lbfgs_pts)
        if e < best_energy:
            best_energy = e
            best_pts = lbfgs_pts
    
    # Multiple restarts with diverse perturbations
    n_restarts = 16
    
    for r in range(n_restarts):
        if remaining() < 20.0:
            break
        
        # Create perturbed starting point with varying scales
        if r == 0:
            start = best_pts.copy()
        elif r < 5:
            # Small perturbations of best
            scale = 0.02 + 0.01 * r
            start = best_pts + scale * rng.standard_normal((N, 3))
            start /= np.linalg.norm(start, axis=1, keepdims=True)
        elif r < 10:
            # Medium perturbations
            scale = 0.08 + 0.02 * (r - 5)
            start = best_pts + scale * rng.standard_normal((N, 3))
            start /= np.linalg.norm(start, axis=1, keepdims=True)
        else:
            # Random restarts with different seeds
            start = _random_sphere(N, rng)
        
        # L-BFGS-B optimization
        time_budget = min(18.0, remaining() - 15.0)
        if time_budget > 4.0:
            opt_pts = _lbfgs_optimize(start, max_iter=1000, time_budget=time_budget, gtol=1e-11)
            e = _log_energy(opt_pts)
            if e < best_energy:
                best_energy = e
                best_pts = opt_pts
    
    # High-precision polish: L-BFGS-B with very tight tolerance
    if remaining() > 12.0:
        polished_lbfgs = _lbfgs_optimize(best_pts, max_iter=3000, time_budget=min(10.0, remaining() - 5.0), gtol=1e-14)
        e = _log_energy(polished_lbfgs)
        if e < best_energy:
            best_energy = e
            best_pts = polished_lbfgs
    
    # Final gradient descent polish with adaptive steps
    if remaining() > 6.0:
        final_gd = _gradient_descent(best_pts, lr_init=0.001, max_iter=8000, tol=1e-16)
        e = _log_energy(final_gd)
        if e < best_energy:
            best_energy = e
            best_pts = final_gd
    
    # One more L-BFGS-B pass if time allows
    if remaining() > 4.0:
        last_lbfgs = _lbfgs_optimize(best_pts, max_iter=2000, time_budget=min(3.5, remaining() - 1.0), gtol=1e-14)
        e = _log_energy(last_lbfgs)
        if e < best_energy:
            best_energy = e
            best_pts = last_lbfgs
    
    # Ensure valid output
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
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
