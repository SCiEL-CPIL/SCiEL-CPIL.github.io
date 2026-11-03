"""Logarithmic energy minimization for N points on the unit sphere.
Uses multi-start L-BFGS-B optimization with diverse initializations including Sobol sequences."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

N = 100
_DEADLINE = 145.0  # safety margin from 150s limit
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _project(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n


def _energy(pts):
    """Fast energy computation using matrix operations."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    iu = np.triu_indices(n, k=1)
    return float(-np.sum(np.log(dist[iu])))


def _energy_grad(pts):
    """Compute energy and gradient (projected onto tangent space).
    Returns (energy, gradient) where gradient is the true gradient of the energy."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    
    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(dist[iu])))
    
    # Gradient of E = sum log(1/d) = -sum log(d)
    # dE/dx_i = - sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # sum_j W_ij * (x_i - x_j) = x_i * sum_j W_ij - sum_j W_ij * x_j
    sum_W = W.sum(axis=1, keepdims=True)
    Wx = W @ pts
    raw_grad = sum_W * pts - Wx
    
    # raw_grad is sum (x_i - x_j)/d^2
    # True gradient is -raw_grad
    grad = -raw_grad
    
    # Project onto tangent space: grad - (grad . x_i) x_i
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    
    return energy, grad


def _precise_energy(pts):
    """Compute energy using math.fsum for accuracy."""
    n = pts.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(rng):
    v = rng.standard_normal((N, 3))
    return _project(v)


def _sobol_sphere(n, seed=0):
    """Generate points on sphere using Sobol sequence."""
    # Sobol sequence in 3D
    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    # We need enough points to get a good distribution, then select or map
    # For sphere, we can map uniform cube to sphere.
    # One way: sample direction from uniform distribution on sphere using normal distribution is better,
    # but Sobol gives low discrepancy in cube.
    # Let's use the normal distribution mapping but with Sobol-like structure?
    # Actually, standard practice for low discrepancy on sphere is to use spherical coordinates
    # derived from low discrepancy sequences in [0,1]^2.
    
    # Generate Sobol points in [0,1]^2
    sampler_2d = qmc.Sobol(d=2, scramble=True, seed=seed)
    # Need at least n points. Sobol requires power of 2.
    m = int(2 ** np.ceil(np.log2(n)))
    y = sampler_2d.random(m)
    
    # Map to sphere
    # theta = arccos(1 - 2 * y[:, 0])
    # phi = 2 * pi * y[:, 1]
    # This gives uniform distribution on sphere with low discrepancy.
    theta = np.arccos(1 - 2 * y[:, 0])
    phi = 2 * np.pi * y[:, 1]
    
    x = np.sin(theta) * np.cos(phi)
    y_coord = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    pts = np.column_stack([x, y_coord, z])
    return pts[:n]


def _quick_grad_descent(pts, n_steps=50, lr=0.01):
    """Quick gradient descent with adaptive learning rate to improve starting point."""
    current = pts.copy()
    best_e = _energy(current)
    for step in range(n_steps):
        e, g = _energy_grad(current)
        # Adaptive step size: scale by inverse of gradient norm
        gnorm = np.linalg.norm(g)
        if gnorm < 1e-14:
            break
        step_lr = lr * (1.0 / (1.0 + gnorm))
        new_pts = _project(current - step_lr * g)
        new_e = _energy(new_pts)
        if new_e < best_e:
            current = new_pts
            best_e = new_e
        else:
            # Halve learning rate
            lr *= 0.5
            if lr < 1e-10:
                break
    return current


def _nelder_mead_refine(pts, max_iter=200, tol=1e-12):
    """Nelder-Mead local refinement for final polish."""
    n = pts.shape[0]
    
    def objective(flat):
        p = flat.reshape(n, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        p = p / norms
        return _energy(p)
    
    try:
        x0 = pts.reshape(-1)
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': max_iter,
                'xatol': tol,
                'fatol': tol,
                'adaptive': True,
            }
        )
        new_pts = _project(result.x.reshape(n, 3))
        return new_pts
    except Exception:
        return pts


def _optimize(pts, deadline, max_iter=2000, ftol=1e-15, gtol=1e-12):
    """Run L-BFGS-B optimization from a starting point."""
    n = pts.shape[0]
    
    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        p = p / norms
        e, g = _energy_grad(p)
        return e, g.reshape(-1)
    
    try:
        x0 = pts.reshape(-1)
        # L-BFGS-B is more memory efficient and faster for this problem size
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': max_iter,
                'gtol': gtol,
                'ftol': ftol,
                'maxls': 100,
                'm': 100,
            }
        )
        new_pts = _project(result.x.reshape(n, 3))
        return new_pts
    except Exception:
        return pts


def _valid(X):
    if X.shape != (N, 3) or not np.all(np.isfinite(X)):
        return False
    if np.any(np.linalg.norm(X, axis=1) < 1e-12):
        return False
    d2 = np.einsum('ij,ij->i', X, X)[:, None] + np.einsum('ij,ij->i', X, X)[None, :] \
         - 2.0 * (X @ X.T)
    np.fill_diagonal(d2, np.inf)
    if np.min(d2) < (1e-9) ** 2:
        return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations - focused for more optimization time
    initializations = []
    
    # 1. Sobol sequences (Low discrepancy) - 12 seeds
    for s in range(12):
        initializations.append(_sobol_sphere(N, seed=s))
        
    # 2. Fibonacci spirals with various offsets
    for offset in [0.1, 0.3, 0.5, 0.7, 0.9]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
        
    # 3. Jittered Fibonacci spirals - 6 variants
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(6):
        jitter_scale = 0.02 + 0.01 * s
        jittered = base + jitter_scale * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
        
    # 4. Random points with different seeds - 10 seeds
    for s in range(10):
        initializations.append(_random_sphere(np.random.default_rng(100 + s)))
    
    best_pts = None
    best_energy = float('inf')
    
    # Score all initializations
    scored_inits = []
    for init_pts in initializations:
        if time.time() > deadline - 30.0:
            break
        e_quick = _energy(init_pts)
        scored_inits.append((e_quick, init_pts))
        
    scored_inits.sort(key=lambda x: x[0])
    
    # Optimize top candidates with two-phase approach
    top_candidates = scored_inits[:60]
    
    for idx, (e_quick, init_pts) in enumerate(top_candidates):
        if time.time() > deadline - 25.0:
            break
        
        # Phase 1: Quick gradient descent to improve starting point
        pre_pts = _quick_grad_descent(init_pts, n_steps=40, lr=0.02)
        
        # Phase 2: L-BFGS-B optimization with moderate tolerances
        optimized = _optimize(pre_pts, deadline, max_iter=1500, ftol=1e-14, gtol=1e-11)
        
        e_precise = _precise_energy(optimized)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized.copy()
            
    # Final polish on the global best with very tight tolerances
    if best_pts is not None and time.time() < deadline - 10.0:
        # High-precision polish pass 1: L-BFGS-B with tight tolerance
        refined = _optimize(best_pts, deadline, max_iter=3000, ftol=1e-16, gtol=1e-14)
        e_precise = _precise_energy(refined)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = refined
        
        # Second polish: Nelder-Mead for local refinement
        if time.time() < deadline - 7.0:
            nm_pts = _nelder_mead_refine(best_pts, max_iter=300, tol=1e-13)
            e_nm = _precise_energy(nm_pts)
            if e_nm < best_energy:
                best_energy = e_nm
                best_pts = nm_pts
        
        # Third polish: L-BFGS-B with ultra-tight tolerance
        if time.time() < deadline - 4.0:
            refined2 = _optimize(best_pts, deadline, max_iter=2000, ftol=1e-18, gtol=1e-16)
            e_precise = _precise_energy(refined2)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = refined2
        
        # Fourth ultra-tight polish if time remains
        if time.time() < deadline - 2.0:
            refined3 = _optimize(best_pts, deadline, max_iter=1500, ftol=1e-20, gtol=1e-17)
            e_precise = _precise_energy(refined3)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = refined3
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N, offset=0.5)
    
    # Final normalization
    best_pts = _project(best_pts)
    
    # Final validation
    final_energy = _precise_energy(best_pts)
    if not np.isfinite(final_energy):
        best_pts = _fibonacci_spiral(N, offset=0.5)
        best_pts = _project(best_pts)
    
    return best_pts
