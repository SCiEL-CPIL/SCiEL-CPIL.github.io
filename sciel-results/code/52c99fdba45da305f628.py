import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|xi-xj|)"""
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0) # Avoid log(0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(N, k=1)
    # Sum log(1/d) = -sum(log(d))
    return float(-np.sum(np.log(dist[iu])))

def _energy_grad(pts):
    """Compute energy and gradient (projected onto tangent space)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0)
    
    # Gradient of -log(d) wrt x_i is (x_i - x_j) / d^2
    # Sum over j != i
    with np.errstate(divide='ignore', invalid='ignore'):
        # terms[i, j] = (x_i - x_j) / |x_i - x_j|^2
        terms = diff / dist_sq[:, :, np.newaxis]
    
    # Zero out diagonal
    idx = np.diag_indices(N)
    terms[idx] = 0.0
    
    # Sum over j for each i
    grad = -np.sum(terms, axis=1)
    
    # Project onto tangent space: g = grad - (grad . x) x
    dot = np.sum(grad * pts, axis=1)[:, np.newaxis]
    grad_proj = grad - dot * pts
    
    # Energy
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(N, k=1)
    energy = float(-np.sum(np.log(dist[iu])))
    
    return energy, grad_proj

def _fibonacci_spiral(n, offset=0.0, jitter=0.0, seed=42):
    """Generate points on sphere using Fibonacci spiral with optional offset and jitter."""
    np.random.seed(seed)
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if jitter > 0:
        x += jitter * np.random.randn(n, 3)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / norms
    return x

def _sobol_sphere(n, seed=42):
    """Generate points on sphere using Sobol sequence for quasi-random uniform coverage."""
    from scipy.stats import qmc
    sampler = qmc.Sobol(d=2, seed=seed)
    samples = sampler.random(n)
    theta = 2.0 * np.pi * samples[:, 0]
    cos_phi = 2.0 * samples[:, 1] - 1.0
    sin_phi = np.sqrt(np.maximum(0.0, 1.0 - cos_phi * cos_phi))
    x = np.column_stack([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
    return x

def _project(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _optimize(pts, deadline, maxiter=500):
    """Optimize points using L-BFGS-B with time limit."""
    def objective_and_grad(flat):
        if time.time() > deadline:
            # Return a large value to stop optimization
            return 1e10, np.zeros_like(flat)
        pts_local = _project(flat.reshape(N, 3))
        energy, grad = _energy_grad(pts_local)
        return energy, grad.flatten()
    
    x0 = pts.flatten()
    
    try:
        result = minimize(
            objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': 1e-12, 'disp': False, 'ftol': 1e-15}
        )
        if result.fun < 1e9: # Check if it stopped due to time limit
            return _project(result.x.reshape(N, 3)), float(result.fun)
    except Exception:
        pass
    
    return pts, _energy(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_x = None
    best_energy = float('inf')
    
    # Phase 1: Multi-start optimization
    # We will try different initializations and optimize each
    # If time runs out, we return the best one found
    
    initializations = []
    # Fibonacci spirals with different offsets
    for i in range(5):
        initializations.append(_fibonacci_spiral(N, offset=i * 0.5, seed=i))
    # Sobol sequences with different seeds
    for i in range(3):
        initializations.append(_sobol_sphere(N, seed=100 + i))
    # Random points
    np.random.seed(42)
    for i in range(2):
        pts = np.random.randn(N, 3)
        initializations.append(_project(pts))
    
    # Optimize each initialization
    for i, pts in enumerate(initializations):
        if time.time() > deadline - 10: # Leave 10s for final polish
            break
        
        # Dynamic maxiter: less for later iterations if time is tight
        time_left = deadline - time.time()
        if time_left < 30:
            maxiter = 200
        else:
            maxiter = 500
            
        x_opt, e_opt = _optimize(pts, deadline, maxiter=maxiter)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_x = x_opt.copy()
            
    # Phase 2: Final high-precision polish on the best solution
    if best_x is not None and time.time() < deadline - 1:
        # Do one more L-BFGS run with very tight tolerances
        def final_objective_and_grad(flat):
            pts_local = _project(flat.reshape(N, 3))
            energy, grad = _energy_grad(pts_local)
            return energy, grad.flatten()
        
        x0 = best_x.flatten()
        try:
            result = minimize(
                final_objective_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'gtol': 1e-14, 'disp': False, 'ftol': 1e-16}
            )
            if result.fun < best_energy:
                best_x = _project(result.x.reshape(N, 3))
                best_energy = float(result.fun)
        except Exception:
            pass
    
    # Final check: ensure valid output
    if best_x is None:
        # Fallback to a simple Fibonacci spiral if optimization failed
        best_x = _fibonacci_spiral(N)
        
    best_x = _project(best_x)
    assert best_x.shape == (N, 3)
    assert np.all(np.isfinite(best_x))
    assert np.all(np.linalg.norm(best_x, axis=1) > 1e-9)
    
    return best_x
