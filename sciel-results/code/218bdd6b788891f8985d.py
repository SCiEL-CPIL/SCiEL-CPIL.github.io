"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

_DEADLINE = 150.0  # seconds
_START = None

def _elapsed():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START

def _energy_and_grad(pts):
    """Compute log energy sum_{i<j} log(1/|x_i - x_j|) and gradient on sphere.
    pts: (N,3) array on unit sphere.
    Returns (energy, grad) where grad is the gradient projected to tangent space."""
    n = pts.shape[0]
    # Pairwise distances
    # Use broadcasting for distances
    # d_ij = |x_i - x_j|
    # We want sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    
    # Compute all pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    d = np.sqrt(np.sum(diff * diff, axis=2))  # (N, N)
    np.fill_diagonal(d, 1.0)  # avoid log(0)
    
    # Energy: sum over i<j of -log(d_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    d_upper = d[iu]
    energy = -np.sum(np.log(d_upper))
    
    # Gradient: d/dx_k [sum_{i<j} -log(d_ij)]
    # For pair (i,j): contribution to grad at point i is (x_i - x_j)/d_ij^2 * ... 
    # Actually: d/dx_i [-log(d_ij)] = -(1/d_ij) * d(d_ij)/d(x_i) = -(1/d_ij) * (x_i - x_j)/d_ij = -(x_i - x_j)/d_ij^2
    # Wait: d(d_ij)/d(x_i) = (x_i - x_j)/d_ij
    # So d/dx_i [-log(d_ij)] = -1/d_ij * (x_i - x_j)/d_ij = -(x_i - x_j)/d_ij^2
    # But we want to minimize energy, and energy = sum -log(d_ij)
    # Gradient of energy w.r.t. x_i: sum_{j != i} -(x_i - x_j)/d_ij^2
    # = sum_{j != i} (x_j - x_i)/d_ij^2
    
    # Let's compute this properly
    # grad_i = sum_{j != i} (x_j - x_i) / d_ij^2
    # But we need to project to tangent space: grad_tan_i = grad_i - (grad_i . x_i) x_i
    
    # Compute matrix of 1/d_ij^2 for i != j
    inv_d2 = np.zeros_like(d)
    mask = ~np.eye(n, dtype=bool)
    inv_d2[mask] = 1.0 / d[mask] ** 2
    
    # grad_i = sum_j (x_j - x_i) * inv_d2[i,j]
    # = (sum_j x_j * inv_d2[i,j]) - x_i * (sum_j inv_d2[i,j])
    row_sums = np.sum(inv_d2, axis=1)  # (N,)
    # sum_j x_j * inv_d2[i,j] = inv_d2 @ pts.T
    weighted_sum = inv_d2 @ pts.T  # (N, 3)
    grad = weighted_sum - np.outer(row_sums, pts)  # (N, 3)
    
    # Project to tangent space
    grad_dot_pts = np.sum(grad * pts, axis=1)  # (N,)
    grad = grad - np.outer(grad_dot_pts, pts)
    
    return energy, grad

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _fibonacci_sphere(n):
    """Generate initial Fibonacci spiral points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=None):
    """Generate random points on unit sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pts = rng.randn(n, 3)
    return _project_to_sphere(pts)

def _lbfgs_local_min(pts, maxiter=1000, ftol=1e-12, gtol=1e-10):
    """Run L-BFGS on the sphere using parameterization or projected gradient."""
    # Use scipy minimize with a parameterization
    # We'll use the fact that we can do line search in tangent space
    # Simpler: use scipy minimize with bounds or just use the gradient
    
    # Actually, let's use a different approach: parameterize each point by its 
    # tangent coordinates. But that's complex.
    
    # Alternative: use scipy.optimize.minimize with method='L-BFGS-B' on flattened 
    # coordinates, but that doesn't respect the sphere constraint.
    
    # Better: use a custom approach where we do gradient descent in tangent space
    # with line search.
    
    best_pts = pts.copy()
    best_energy, _ = _energy_and_grad(best_pts)
    
    for it in range(maxiter):
        if _elapsed() > _DEADLINE - 10:
            break
        energy, grad = _energy_and_grad(best_pts)
        if energy < best_energy:
            best_energy = energy
            # best_pts = best_pts.copy()  # already is
        if np.max(np.abs(grad)) < gtol:
            break
        
        # Line search in tangent space
        # Step: x_new = (x - alpha * grad) / |x - alpha * grad|
        # Try a few step sizes
        best_alpha = 1e-3
        best_e = best_energy
        for alpha in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
            cand = best_pts - alpha * grad
            cand = _project_to_sphere(cand)
            e, _ = _energy_and_grad(cand)
            if e < best_e:
                best_e = e
                best_alpha = alpha
        
        if best_e >= best_energy:
            # Try smaller steps
            for alpha in [1e-6, 1e-7, 1e-8]:
                cand = best_pts - alpha * grad
                cand = _project_to_sphere(cand)
                e, _ = _energy_and_grad(cand)
                if e < best_e:
                    best_e = e
                    best_alpha = alpha
            if best_e >= best_energy:
                break
        
        best_pts = _project_to_sphere(best_pts - best_alpha * grad)
    
    return best_pts, best_energy

def _basin_hopping(pts, n_restarts=5, max_iters=50, step_scale=0.1):
    """Basin hopping: perturb, then local minimize."""
    best_pts = pts.copy()
    best_energy, _ = _energy_and_grad(best_pts)
    
    for r in range(n_restarts):
        if _elapsed() > _DEADLINE - 15:
            break
        # Perturb
        noise = np.random.randn(*pts.shape) * step_scale
        perturbed = _project_to_sphere(pts + noise)
        
        # Local minimize
        cand, e = _lbfgs_local_min(perturbed, maxiter=200, ftol=1e-12, gtol=1e-10)
        if e < best_energy:
            best_energy = e
            best_pts = cand
    
    return best_pts, best_energy

def construct():
    global _START
    _START = time.time()
    
    # Start with Fibonacci spiral
    pts = _fibonacci_sphere(N)
    best_pts, best_energy = _lbfgs_local_min(pts, maxiter=500, ftol=1e-14, gtol=1e-12)
    
    # Basin hopping with multiple restarts
    for i in range(8):
        if _elapsed() > _DEADLINE - 20:
            break
        # Perturb from current best
        noise = np.random.randn(*best_pts.shape) * 0.05
        perturbed = _project_to_sphere(best_pts + noise)
        cand, e = _lbfgs_local_min(perturbed, maxiter=300, ftol=1e-14, gtol=1e-12)
        if e < best_energy:
            best_energy = e
            best_pts = cand
    
    # Final polish with very tight tolerance
    if _elapsed() < _DEADLINE - 5:
        final_pts, final_energy = _lbfgs_local_min(best_pts, maxiter=1000, ftol=1e-16, gtol=1e-14)
        if final_energy < best_energy:
            best_pts = final_pts
            best_energy = final_energy
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    if not np.all(np.isfinite(best_pts)):
        # Fallback to Fibonacci
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
# EVOLVE-BLOCK-END
