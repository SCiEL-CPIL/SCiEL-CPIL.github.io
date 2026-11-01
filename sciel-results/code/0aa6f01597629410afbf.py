"""Combined logarithmic energy minimization for 100 points on the sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log(|xi-xj|).
    Uses math.fsum for accuracy as specified in the task."""
    n = pts.shape[0]
    dots = pts @ pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-15)
    dist = np.sqrt(d2)
    iu = np.triu_indices(n, k=1)
    d_ij = dist[iu]
    terms = [-math.log(d) for d in d_ij]
    return math.fsum(terms)

def _gradient_tangent(pts):
    """Tangent-space gradient of logarithmic energy on the sphere.
    Energy E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    dE/dx_i = sum_{j!=i} (x_i - x_j) / d_ij^2
    Project onto tangent space: grad_tan = grad - (grad . x_i) x_i
    """
    n = pts.shape[0]
    dots = pts @ pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-15)
    
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad_i = sum_j (x_i - x_j) / d_ij^2 = x_i * sum_j(1/d_ij^2) - sum_j x_j / d_ij^2
    sum_inv_d2 = inv_d2.sum(axis=1)  # (n,)
    sum_xj_inv_d2 = (pts.T @ inv_d2).T  # (n, 3)
    
    grad = pts * sum_inv_d2[:, np.newaxis] - sum_xj_inv_d2  # (n, 3)
    
    # Project to tangent space
    grad_dot_xi = np.sum(grad * pts, axis=1, keepdims=True)  # (n, 1)
    grad_tan = grad - grad_dot_xi * pts
    
    return grad_tan

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _fibonacci_spiral(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, seed=42):
    """Generate n random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _perturbed_points(base_pts, seed=0, sigma=0.03):
    """Generate perturbed version of base points."""
    rng = np.random.RandomState(seed)
    noise = rng.randn(*base_pts.shape) * sigma
    return _project_to_sphere(base_pts + noise)

def _lbfgs_tangent(pts, max_iter=2000, ftol=1e-12, gtol=1e-10, maxls=20):
    """L-BFGS-B optimization using tangent-space gradient.
    We optimize in R^3 but use the tangent gradient and project back.
    """
    x0 = pts.flatten()
    
    def fun(x):
        p = x.reshape(N, 3)
        p = _project_to_sphere(p)
        e = _log_energy(p)
        g = _gradient_tangent(p)
        return e, g.flatten()
    
    best_pts = _project_to_sphere(pts.copy())
    best_energy = _log_energy(best_pts)
    
    try:
        res = minimize(fun, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'maxls': maxls})
        if np.isfinite(res.fun) and res.fun < best_energy:
            new_pts = _project_to_sphere(res.x.reshape(N, 3))
            e_new = _log_energy(new_pts)
            if e_new < best_energy:
                best_pts = new_pts
                best_energy = e_new
    except Exception:
        pass
    
    return best_pts, best_energy

def _trust_constr_optimize(pts, max_iter=500, deadline=None):
    """Trust-constr optimization with sphere constraint for higher precision."""
    from scipy.optimize import LinearConstraint
    
    x0 = pts.flatten()
    
    def fun(x):
        p = x.reshape(N, 3)
        p = _project_to_sphere(p)
        return _log_energy(p)
    
    def jac(x):
        p = x.reshape(N, 3)
        p = _project_to_sphere(p)
        return _gradient_tangent(p).flatten()
    
    # Constraint: ||x_i||^2 = 1 for each point
    # This is hard to express as linear constraint, so use inequality
    # Actually, trust-constr can handle nonlinear constraints
    def constraint_fun(x):
        p = x.reshape(N, 3)
        return np.sum(p * p, axis=1) - 1.0  # should be 0
    
    # Use trust-constr with the sphere as a constraint
    # But this is complex. Let's just do a final polish with tight L-BFGS
    return pts, _log_energy(pts)

def _riemannian_descent(pts, max_iter=1000, tol=1e-10, deadline=None):
    """Riemannian gradient descent with backtracking line search on the sphere."""
    pts = _project_to_sphere(pts.copy())
    f = _log_energy(pts)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        grad = _gradient_tangent(pts)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # Adaptive initial step size
        step = 1.0 / (1.0 + 0.5 * grad_norm)
        
        accepted = False
        for ls_iter in range(60):
            new_pts = _project_to_sphere(pts - step * grad)
            new_f = _log_energy(new_pts)
            
            if np.isfinite(new_f) and new_f < f:
                # Accept if sufficient decrease (Armijo condition)
                if new_f < f - 1e-4 * step * grad_norm * grad_norm:
                    accepted = True
                    break
            step *= 0.8
        
        if accepted:
            pts = new_pts
            f = new_f
        else:
            break
    
    return pts, f

def _optimize_pipeline(pts0, deadline):
    """Multi-stage optimization: L-BFGS -> Riemannian -> tight L-BFGS polish."""
    best_pts = _project_to_sphere(pts0.copy())
    best_energy = _log_energy(best_pts)
    
    # Stage 1: L-BFGS with moderate tolerance
    if time.time() < deadline - 10.0:
        pts, e = _lbfgs_tangent(best_pts, max_iter=2000, ftol=1e-12, gtol=1e-10)
        if e < best_energy:
            best_pts = pts
            best_energy = e
    
    # Stage 2: Riemannian gradient descent for robustness
    if time.time() < deadline - 8.0:
        pts, e = _riemannian_descent(best_pts, max_iter=500, tol=1e-12, deadline=deadline - 8.0)
        if e < best_energy:
            best_pts = pts
            best_energy = e
    
    # Stage 3: Tight L-BFGS polish
    if time.time() < deadline - 3.0:
        pts, e = _lbfgs_tangent(best_pts, max_iter=3000, ftol=1e-14, gtol=1e-12, maxls=30)
        if e < best_energy:
            best_pts = pts
            best_energy = e
    
    return best_pts, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s
    
    # Start with Fibonacci spiral
    pts = _fibonacci_spiral(N)
    best_pts, best_energy = _optimize_pipeline(pts, deadline)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 40.0:
        return _project_to_sphere(best_pts)
    
    # Try diverse random restarts
    seeds_to_try = [1, 2, 3, 7, 13, 42, 100, 123, 456, 789, 1000, 2024, 31415, 27182, 16180]
    for seed in seeds_to_try:
        if time.time() > deadline - 15.0:
            break
        
        # Vary the perturbation level for diversity
        sigma = 0.02 + 0.01 * (seed % 5)
        pts_init = _perturbed_points(_fibonacci_spiral(N), seed=seed, sigma=sigma)
        
        pts, e = _optimize_pipeline(pts_init, deadline)
        if e < best_energy:
            best_pts = pts
            best_energy = e
    
    # Try pure random starts with smaller budget
    if time.time() < deadline - 12.0:
        for seed in [999, 12345, 54321, 77777, 88888]:
            if time.time() > deadline - 8.0:
                break
            pts_init = _random_points(N, seed=seed)
            pts, e = _optimize_pipeline(pts_init, deadline)
            if e < best_energy:
                best_pts = pts
                best_energy = e
    
    # Final high-precision polish on the best configuration
    if time.time() < deadline - 5.0:
        # Multiple rounds of tight L-BFGS and Riemannian for convergence
        for round_num in range(3):
            if time.time() > deadline - 3.0:
                break
            pts, e = _lbfgs_tangent(best_pts, max_iter=3000, ftol=1e-15, gtol=1e-14, maxls=50)
            if e < best_energy:
                best_pts = pts
                best_energy = e
            
            if time.time() > deadline - 2.0:
                break
            pts, e = _riemannian_descent(best_pts, max_iter=500, tol=1e-15, deadline=deadline - 2.0)
            if e < best_energy:
                best_pts = pts
                best_energy = e
    
    # Final normalization and validation
    best_pts = _project_to_sphere(best_pts)
    
    # Ensure no points are too close
    dots = best_pts @ best_pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    min_d2 = d2.min()
    if min_d2 < 1e-18:
        # Points too close, add small perturbation
        rng = np.random.RandomState(12345)
        best_pts = _project_to_sphere(best_pts + rng.randn(N, 3) * 1e-6)
    
    return best_pts

# EVOLVE-BLOCK-END
