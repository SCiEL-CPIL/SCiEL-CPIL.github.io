"""Minimize logarithmic energy of 100 points on the unit sphere."""
import numpy as np
import math
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math

N = 100


def _energy_and_grad(pts):
    """Compute energy and gradient of log-energy on the sphere.
    
    Energy E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(d_ij)
    
    Gradient of E w.r.t. x_i (in R^3):
    dE/dx_i = sum_{j != i} (x_i - x_j) / d_ij^2
    
    This is the gradient in ambient R^3. For optimization on the sphere,
    we can project or just use it directly with projection back.
    """
    # Compute pairwise differences
    diff = pts[:, None, :] - pts[None, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    # Energy: -sum_{i<j} log(d_ij) = -0.5 * sum_{i != j} log(d_ij)
    # Since sum over i<j of log(d_ij) = 0.5 * sum_{i != j} log(d_ij)
    # E = -0.5 * sum_{i != j} log(d_ij) = -0.25 * sum_{i != j} log(d_ij^2)
    # log(d_ij) = 0.5 * log(d_ij^2)
    # E = -sum_{i<j} 0.5 * log(d_ij^2) = -0.25 * sum_{i != j} log(d_ij^2)
    
    # Actually let's just compute it directly for the upper triangle
    iu = np.triu_indices(N, k=1)
    d_sq = dist_sq[iu]
    d_sq = np.clip(d_sq, 1e-30, None)
    
    # E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij) = -0.5 * sum_{i<j} log(d_ij^2)
    energy = -0.5 * np.sum(np.log(d_sq))
    
    # Gradient: dE/dx_i = sum_{j != i} (x_i - x_j) / d_ij^2
    # = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # 
    # For each i: grad_i = sum_j (x_i - x_j) / d_ij^2
    # 
    # We can compute this as: for each pair (i,j), add (x_i - x_j)/d_ij^2 to grad_i
    # and (x_j - x_i)/d_ij^2 to grad_j
    # 
    # Vectorized: grad[i] = sum_j (x_i - x_j) / d_ij^2
    # = x_i * sum_j 1/d_ij^2 - sum_j x_j / d_ij^2
    
    inv_d_sq = 1.0 / dist_sq  # (N, N)
    # Set diagonal to 0 to avoid self-interaction
    np.fill_diagonal(inv_d_sq, 0.0)
    
    # grad[i] = sum_j (x_i - x_j) * inv_d_sq[i,j]
    # = x_i * sum_j inv_d_sq[i,j] - sum_j x_j * inv_d_sq[i,j]
    
    sum_inv = np.sum(inv_d_sq, axis=1)  # (N,)
    # sum_j x_j * inv_d_sq[i,j] = (inv_d_sq @ pts.T).T
    weighted_sum = inv_d_sq @ pts.T  # (3, N)
    weighted_sum_T = weighted_sum.T  # (N, 3)
    
    grad = pts * sum_inv[:, None] - weighted_sum_T
    
    return energy, grad


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms


def _fibonacci_sphere(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=None):
    """Generate n random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def _optimize_from_start(start_pts, deadline, max_iter=3000):
    """Run L-BFGS optimization on the sphere, respecting deadline."""
    x0 = start_pts.flatten().copy()
    
    def objective(x):
        if time.time() > deadline:
            return 1e10
        pts = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_grad(pts)
        # Project gradient to tangent space of sphere
        # tangent gradient = g - (g . x) * x
        g_tangent = g - np.sum(g * pts, axis=1, keepdims=True) * pts
        return e, g_tangent.flatten()
    
    try:
        res = minimize(
            objective, x0,
            jac=True,
            method='L-BFGS-B',
            options={
                'maxiter': max_iter,
                'maxfun': 100000,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxls': 50
            }
        )
        result_pts = _project_to_sphere(res.x.reshape(N, 3))
        return result_pts
    except Exception:
        return None


def _local_refine(pts, deadline, n_iters=50):
    """Local refinement with small random perturbations and re-optimization."""
    best_pts = pts.copy()
    best_energy, _ = _energy_and_grad(best_pts)
    
    for _ in range(n_iters):
        if time.time() > deadline - 5:
            break
        
        # Small random perturbation
        noise_scale = 0.05
        noise = np.random.randn(N, 3) * noise_scale
        new_pts = _project_to_sphere(best_pts + noise)
        
        if not np.all(np.isfinite(new_pts)):
            continue
        
        # Quick energy check
        e_new, _ = _energy_and_grad(new_pts)
        if e_new < best_energy:
            best_energy = e_new
            best_pts = new_pts
    
    return best_pts


def construct():
    """Main entry point: generate candidates, optimize, keep best."""
    start_time = time.time()
    # Reserve 15 seconds for safety before 150s limit
    deadline = start_time + 135.0
    
    # Best solution tracking
    best_pts = None
    best_energy = float('inf')
    
    np.random.seed(42)
    
    # Strategy 1: Fibonacci spiral (good initial structure)
    if time.time() < deadline - 30:
        fib_pts = _fibonacci_sphere(N)
        result = _optimize_from_start(fib_pts, deadline - 10)
        if result is not None and np.all(np.isfinite(result)):
            e, _ = _energy_and_grad(result)
            if e < best_energy:
                best_energy = e
                best_pts = result
    
    # Strategy 2: Multiple random restarts with L-BFGS
    n_restarts = 15
    for i in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        seed = 100 + i
        random_pts = _random_sphere(N, seed=seed)
        result = _optimize_from_start(random_pts, deadline - 10, max_iter=2000)
        if result is not None and np.all(np.isfinite(result)):
            e, _ = _energy_and_grad(result)
            if e < best_energy:
                best_energy = e
                best_pts = result
    
    # Strategy 3: Local refinement on the best solution
    if best_pts is not None and time.time() < deadline - 10:
        refined = _local_refine(best_pts, deadline - 5, n_iters=30)
        e_ref, _ = _energy_and_grad(refined)
        if e_ref < best_energy:
            best_energy = e_ref
            best_pts = refined
    
    # Final validation
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    
    # Check for any issues
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_sphere(N)
    
    # Verify minimum distance
    diff = best_pts[:, None, :] - best_pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    min_dist = np.min(dist[iu])
    
    if min_dist < 1e-9:
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
# EVOLVE-BLOCK-END
