import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150

def _log_energy_fast(pts):
    """Vectorized log energy computation for speed using squared distances."""
    n = len(pts)
    # Using dot products for squared distances is faster
    # d^2 = 2 - 2 * dot(x_i, x_j) for unit vectors
    dots = pts @ pts.T
    dists_sq = 2.0 - 2.0 * dots
    # Numerical stability
    dists_sq = np.maximum(dists_sq, 1e-15)
    
    iu = np.triu_indices(n, k=1)
    d2 = dists_sq[iu]
    
    # log(1/d) = -0.5 * log(d^2)
    return -0.5 * np.sum(np.log(d2))

def _log_energy_grad(pts):
    """Compute gradient of logarithmic energy w.r.t. points on sphere (projected)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad[i, k] = sum_j (-(pts[i,k] - pts[j,k]) / dists_sq[i,j])
    # = -pts[i,k] * sum_j(1/d2) + sum_j(pts[j,k] / d2)
    weighted_sum_pts = inv_d2 @ pts
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts
    
    # Project gradient to tangent space
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    return grad

def _energy_and_grad_3d(x):
    """Energy and gradient for L-BFGS-B on 3D coordinates."""
    pts = x.reshape(N, 3)
    energy = _log_energy_fast(pts)
    grad = _log_energy_grad(pts)
    return energy, grad.flatten()

def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_energy = float('inf')
    best_pts = None
    
    # Generate initializations
    initializations = []
    
    # 1. Fibonacci Sphere
    fib_pts = _fibonacci_sphere(N)
    initializations.append(fib_pts.copy())
    
    # 2. Random uniform on sphere
    for seed in range(4):
        rng_local = np.random.default_rng(seed * 123 + 7)
        pts_rand = rng_local.standard_normal((N, 3))
        pts_rand = _normalize(pts_rand)
        initializations.append(pts_rand)
    
    # 3. Perturbed Fibonacci
    for seed in range(3):
        rng_local = np.random.default_rng(seed * 456 + 13)
        pts_pert = fib_pts.copy() + rng_local.standard_normal((N, 3)) * 0.02
        pts_pert = _normalize(pts_pert)
        initializations.append(pts_pert)
    
    # 4. Slightly different Fibonacci variants
    for offset in [0.3, 0.7]:
        i = np.arange(N) + offset
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        initializations.append(_normalize(pts))
    
    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 20:
            break
        
        x0 = init_pts.flatten()
        
        try:
            res = minimize(
                _energy_and_grad_3d,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 2000,
                    'ftol': 1e-14,
                    'gtol': 1e-10,
                    'maxls': 50
                }
            )
            
            current_pts = _normalize(res.x.reshape(N, 3))
            current_energy = _log_energy_fast(current_pts)
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = current_pts.copy()
                
        except Exception:
            pass
    
    if best_pts is None:
        best_pts = initializations[0].copy()
        best_energy = _log_energy_fast(best_pts)
    
    # Final high-precision polish with smaller steps
    if time.time() < deadline - 5:
        current_pts = best_pts.copy()
        current_energy = best_energy
        
        # Use a very tight L-BFGS-B polish
        try:
            x0 = current_pts.flatten()
            res = minimize(
                _energy_and_grad_3d,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 5000,
                    'ftol': 1e-16,
                    'gtol': 1e-12,
                    'maxls': 100
                }
            )
            polished_pts = _normalize(res.x.reshape(N, 3))
            polished_energy = _log_energy_fast(polished_pts)
            
            if polished_energy < current_energy:
                best_energy = polished_energy
                best_pts = polished_pts
        except Exception:
            pass
    
    # Final normalization
    best_pts = _normalize(best_pts)
    
    return best_pts
