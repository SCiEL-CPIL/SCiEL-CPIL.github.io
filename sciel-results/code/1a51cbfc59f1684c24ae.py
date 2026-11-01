"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log(|xi-xj|)"""
    # pts: (N, 3) array on unit sphere
    # Compute pairwise distances
    n = pts.shape[0]
    # Use squared distances for efficiency, then sqrt
    # dist^2[i,j] = |xi|^2 + |xj|^2 - 2*xi.xj = 2 - 2*xi.xj (since unit)
    dots = pts @ pts.T  # (n, n)
    # dist^2 = 2 - 2*dots, clip to avoid negative due to fp
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)  # avoid log(0)
    # Clip small values
    d2 = np.maximum(d2, 1e-15)
    dist = np.sqrt(d2)
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    log_d = np.log(dist[iu])
    # Energy = -sum log(dist) = sum log(1/dist)
    return -math.fsum(log_d)

def _gradient_log_energy(pts):
    """Gradient of logarithmic energy w.r.t. points on sphere.
    
    E = -sum_{i<j} log(|xi - xj|)
    dE/dxi = sum_{j != i} (xi - xj) / |xi - xj|^2  (but only for j > i in the pair, 
    so we need to be careful with the factor)
    
    Actually: E = sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log(|xi-xj|)
    dE/dxi = sum_{j>i} (xi - xj)/|xi-xj|^2 + sum_{j<i} (xi - xj)/|xi-xj|^2
           = sum_{j != i} (xi - xj) / |xi - xj|^2
    
    But this is the gradient in R^3. For constrained optimization on the sphere,
    we project onto the tangent space: grad_tangent = grad - (grad . xi) * xi
    """
    n = pts.shape[0]
    grad = np.zeros_like(pts)
    # Compute all pairwise
    dots = pts @ pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-15)
    dist = np.sqrt(d2)
    
    # grad_i = sum_{j != i} (xi - xj) / |xi - xj|^2
    # (xi - xj) / |xi - xj|^2 = (xi - xj) / d2_ij
    # We can compute: for each i, sum over j != i of (xi - xj)/d2_ij
    # = xi * sum_{j!=i} 1/d2_ij - sum_{j!=i} xj / d2_ij
    
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    # sum_{j!=i} 1/d2_ij
    sum_inv_d2 = inv_d2.sum(axis=1)  # (n,)
    
    # sum_{j!=i} xj / d2_ij
    # = pts.T @ inv_d2  -> (3, n) -> transpose to (n, 3)
    sum_xj_inv_d2 = (pts.T @ inv_d2).T  # (n, 3)
    
    grad = pts * sum_inv_d2[:, np.newaxis] - sum_xj_inv_d2
    
    # Project onto tangent space: grad_tan = grad - (grad . xi) * xi
    grad_dot_xi = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tan = grad - grad_dot_xi * pts
    
    return grad_tan

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / np.maximum(norms, 1e-15)

def _perturb_sphere(pts, rng, sigma=0.05):
    """Perturb points and renormalize."""
    noise = rng.standard_normal(pts.shape) * sigma
    new_pts = pts + noise
    return _project_to_sphere(new_pts)

def _optimize_from_start(pts0, time_budget):
    """Optimize from a starting configuration using L-BFGS-B with projection.
    
    Uses a parameterization: optimize in R^3 and project to sphere after each step.
    Alternatively, use a local coordinate system. Here we use the simple approach
    of optimizing in R^3 and projecting.
    """
    best_pts = _project_to_sphere(pts0.copy())
    best_energy = _log_energy(best_pts)
    
    t0 = time.time()
    elapsed = 0.0
    
    # Use L-BFGS-B with a penalty-free approach: optimize in R^3, project
    # Actually, better to use the tangent space gradient with a retraction-based method
    # For simplicity, use scipy minimize with the R^3 gradient and projection
    
    # Let's use a simple gradient descent with backtracking in R^3, projecting each step
    # This is more robust than L-BFGS for constrained problems
    
    # Actually, let's use scipy's L-BFGS-B on the R^3 coordinates with the projected gradient
    # The trick: define a function that takes R^3 coords, projects to sphere, returns energy
    # and the gradient is the projected gradient pulled back
    
    def objective_and_grad(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        e = _log_energy(pts)
        g = _gradient_log_energy(pts)
        # The gradient w.r.t. x (in R^3) when we have pts = x/|x|
        # d(pts_i)/dx = (I - pts_i pts_i^T) / |x_i|
        # So grad_x = (I - pts_i pts_i^T) * grad_pts_i / |x_i|
        # But since we're optimizing over x and projecting, the chain rule gives:
        # grad_x_i = (I - pts_i pts_i^T) @ grad_pts_i / |x_i|
        # Actually for x -> pts = x/|x|, dpts/dx = (I - pts pts^T)/|x|
        # So grad_x = (I - pts pts^T) @ grad_pts / |x|
        norms = np.linalg.norm(x.reshape(N, 3), axis=1)
        norms = np.maximum(norms, 1e-15)
        # (I - pts_i pts_i^T) @ grad_pts_i = grad_tan (already computed)
        grad_x = g / norms[:, np.newaxis]
        return e, grad_x.reshape(-1)
    
    # Use L-BFGS-B
    x0 = best_pts.reshape(-1)
    e0, g0 = objective_and_grad(x0)
    
    # Try L-BFGS-B
    try:
        res = minimize(
            lambda x: objective_and_grad(x)[0],
            x0,
            jac=lambda x: objective_and_grad(x)[1],
            method='L-BFGS-B',
            options={'maxiter': 500, 'gtol': 1e-12, 'maxfun': 1000, 'ftol': 1e-14}
        )
        if res.success or res.fun < best_energy:
            pts_new = _project_to_sphere(res.x.reshape(N, 3))
            e_new = _log_energy(pts_new)
            if e_new < best_energy:
                best_pts = pts_new
                best_energy = e_new
    except Exception:
        pass
    
    elapsed = time.time() - t0
    
    # If we have time left, do a polish pass with tighter tolerance
    if time.time() - t0 < time_budget * 0.5:
        try:
            x1 = best_pts.reshape(-1)
            res2 = minimize(
                lambda x: objective_and_grad(x)[0],
                x1,
                jac=lambda x: objective_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 1000, 'gtol': 1e-15, 'maxfun': 2000, 'ftol': 1e-16}
            )
            if res2.fun < best_energy:
                pts_new = _project_to_sphere(res2.x.reshape(N, 3))
                e_new = _log_energy(pts_new)
                if e_new < best_energy:
                    best_pts = pts_new
                    best_energy = e_new
        except Exception:
            pass
    
    return best_pts, best_energy

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # Fibonacci spiral
    initializations.append(_fibonacci_sphere(N))
    
    # Random starts
    for _ in range(5):
        initializations.append(_random_sphere(N, rng))
    
    # Perturbed Fibonacci
    for sigma in [0.02, 0.05, 0.1, 0.2]:
        initializations.append(_perturb_sphere(_fibonacci_sphere(N), rng, sigma))
    
    # More random
    for _ in range(5):
        initializations.append(_random_sphere(N, rng))
    
    best_pts = initializations[0]
    best_energy = _log_energy(best_pts)
    
    # Try each initialization
    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 10.0:
            break
        
        remaining = deadline - time.time()
        # Allocate time proportional to remaining, with a minimum
        time_for_this = min(remaining * 0.8, 30.0)
        
        pts, energy = _optimize_from_start(init_pts, time_for_this)
        
        if energy < best_energy:
            best_pts = pts
            best_energy = energy
    
    # Final polish on the best result
    if time.time() < deadline - 5.0:
        try:
            pts_final, e_final = _optimize_from_start(best_pts, deadline - time.time() - 2.0)
            if e_final < best_energy:
                best_pts = pts_final
                best_energy = e_final
        except Exception:
            pass
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    
    # Validate
    if not np.all(np.isfinite(best_pts)):
        return _fibonacci_sphere(N)
    
    return best_pts
# EVOLVE-BLOCK-END
