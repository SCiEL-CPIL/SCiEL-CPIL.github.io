"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
from math import log, fsum
import time
from scipy.optimize import minimize

DEADLINE = 120.0
SAFETY_MARGIN = 0.90  # use 90% of time

def _log_energy_precise(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using fsum."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return 1e30
            terms.append(-log(d))
    return fsum(terms)

def _fibonacci_spiral(n, offset=0.0):
    """Golden-angle spiral points on unit sphere with optional phase offset."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, rng):
    """Random points on unit sphere."""
    v = rng.standard_normal((n, 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def _energy_and_grad_cartesian(pts):
    """
    Compute logarithmic energy and its gradient w.r.t. Cartesian coordinates.
    Energy = sum_{i<j} -log(|x_i - x_j|)
    grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    # Pairwise differences: diff[i,j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    # Squared distances
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)
    
    # Inverse squared distances
    inv_r2 = 1.0 / r2
    np.fill_diagonal(inv_r2, 0.0)
    
    # Gradient: grad_i = sum_j (x_i - x_j) / |x_i - x_j|^2
    grad = np.sum(diff * inv_r2[:, :, np.newaxis], axis=1)
    
    # Energy: sum_{i<j} -log(|x_i - x_j|) = -0.5 * sum_{i!=j} -log(|x_i - x_j|)
    r = np.sqrt(r2)
    np.fill_diagonal(r, 1.0)
    log_r = np.log(r)
    np.fill_diagonal(log_r, 0.0)
    energy = 0.5 * np.sum(log_r)
    
    return energy, grad

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _lbfgs_optimize_spherical(pts, max_time=30.0):
    """
    Optimize points on the sphere using L-BFGS-B with analytic gradient.
    We parameterize by Cartesian coordinates and project after each step,
    but use the projected gradient for the optimizer.
    
    Actually, a better approach: use scipy.optimize.minimize with the
    constraint that each point has unit norm. We'll use the SLSQP method
    or reformulate using spherical coordinates.
    
    For simplicity and robustness, we'll use a custom approach:
    - Flatten the points to a 1D vector
    - Define objective and gradient that work on the flattened vector
    - Use L-BFGS-B with the constraint that each point is on the unit sphere
    
    Since L-BFGS-B doesn't handle equality constraints well, we'll use
    a penalty approach or switch to SLSQP.
    
    Actually, the cleanest approach: optimize in Cartesian space with
    projection after each gradient evaluation. L-BFGS-B will work on
    the projected points.
    """
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    start = time.time()
    
    # Flatten for scipy
    x0 = pts.flatten()
    
    def objective(x):
        p = x.reshape(n, 3)
        p = _project_to_sphere(p)
        e, _ = _energy_and_grad_cartesian(p)
        return e
    
    def gradient(x):
        p = x.reshape(n, 3)
        p = _project_to_sphere(p)
        e, g = _energy_and_grad_cartesian(p)
        # Project gradient to tangent space
        g_dot_p = np.sum(g * p, axis=1, keepdims=True)
        g_tan = g - g_dot_p * p
        return g_tan.flatten()
    
    # Use L-BFGS-B
    result = minimize(
        objective,
        x0,
        jac=gradient,
        method='L-BFGS-B',
        options={
            'maxiter': 200,
            'ftol': 1e-12,
            'gtol': 1e-10,
            'maxfun': 500
        }
    )
    
    x_opt = result.x.reshape(n, 3)
    x_opt = _project_to_sphere(x_opt)
    
    # Check validity
    diff = x_opt[:, np.newaxis, :] - x_opt[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-9:
        return pts  # Return original if invalid
    
    return x_opt

def _polish_high_precision(pts, max_iters=100):
    """
    High-precision polish using very small gradient steps with tight tolerance.
    """
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    best_energy = _log_energy_precise(pts)
    
    for it in range(max_iters):
        if time.time() - (time.time() - 0) > 5:  # Safety: don't spend too long
            break
        
        energy, grad = _energy_and_grad_cartesian(pts)
        
        # Project gradient to tangent space
        grad_dot_x = np.sum(grad * pts, axis=1, keepdims=True)
        grad_tan = grad - grad_dot_x * pts
        
        grad_norm = np.linalg.norm(grad_tan)
        if grad_norm < 1e-14:
            break
        
        # Very small step for high precision
        step_size = min(1e-4, 0.5 / (grad_norm + 1e-14))
        
        new_pts = pts - step_size * grad_tan
        new_pts = _project_to_sphere(new_pts)
        
        # Check validity
        diff = new_pts[:, np.newaxis, :] - new_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        if np.min(dists) < 1e-9:
            break
        
        new_energy = _log_energy_precise(new_pts)
        if new_energy < best_energy:
            pts = new_pts
            best_energy = new_energy
        else:
            # Reduce step size further
            step_size *= 0.5
            if step_size < 1e-16:
                break
    
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE * SAFETY_MARGIN
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial configurations
    inits = []
    # Fibonacci spiral variants with different offsets
    for offset in [0.0, 0.3, 0.7, 1.2, 2.0, 3.0]:
        inits.append(_fibonacci_spiral(N, offset=offset))
    # Random starts with different seeds
    for seed in [1, 2, 3, 4, 5]:
        rng_i = np.random.default_rng(seed)
        inits.append(_random_points(N, rng_i))
    
    best_pts = None
    best_energy = 1e30
    
    # Time budget per initialization
    n_inits = len(inits)
    time_per_init = (deadline - start_time - 15) / n_inits  # Leave 15s for final polish
    time_per_init = max(5.0, time_per_init)
    
    for init_idx, init in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > deadline - 15:
            break
        
        # Optimize using L-BFGS-B
        time_budget = min(time_per_init, deadline - 15 - elapsed)
        if time_budget < 3.0:
            break
        
        optimized = _lbfgs_optimize_spherical(init, max_time=time_budget)
        
        # Check validity
        diff = optimized[:, np.newaxis, :] - optimized[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        if min_dist < 1e-9:
            continue
        
        # Calculate precise energy
        e = _log_energy_precise(optimized)
        
        if e < best_energy:
            best_energy = e
            best_pts = optimized.copy()
    
    # Final high-precision polish on the best solution
    if best_pts is not None and time.time() < deadline - 2:
        final_pts = _polish_high_precision(best_pts, max_iters=200)
        
        diff = final_pts[:, np.newaxis, :] - final_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        if min_dist >= 1e-9:
            e_final = _log_energy_precise(final_pts)
            if e_final < best_energy:
                best_energy = e_final
                best_pts = final_pts.copy()
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    # Ensure unit norm
    best_pts = _project_to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
