"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _energy_fsum(pts):
    """Compute logarithmic energy using math.fsum for accuracy."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _energy_and_grad(pts):
    """Compute energy and gradient of logarithmic energy."""
    n = len(pts)
    energy = 0.0
    grad = np.zeros_like(pts)
    
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return float('inf'), grad
            # log(1/d) = -log(d), derivative wrt pts[i]: -(pts[i]-pts[j])/d^2
            # But we want to minimize sum log(1/d) = -sum log(d)
            # d/dx_i [-log(||x_i - x_j||)] = -(x_i - x_j)/||x_i - x_j||^2
            term = -1.0 / d
            diff = pts[i] - pts[j]
            grad[i] += term * diff
            grad[j] -= term * diff
            energy += math.log(1.0 / d)
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _sphere_fun(pts_flat):
    """Objective function for scipy minimize on flattened points."""
    pts = pts_flat.reshape(-1, 3)
    energy, _ = _energy_and_grad(pts)
    return energy


def _sphere_grad(pts_flat):
    """Gradient for scipy minimize on flattened points."""
    pts = pts_flat.reshape(-1, 3)
    _, grad = _energy_and_grad(pts)
    return grad.ravel()


def _fibonacci_spiral():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _optimize_single(pts_init, time_budget):
    """Optimize a single configuration using L-BFGS-B."""
    start_time = time.time()
    pts = pts_init.copy()
    
    # First pass: coarse optimization
    result = minimize(
        _sphere_fun,
        pts.ravel(),
        jac=_sphere_grad,
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    if time.time() - start_time > time_budget:
        return pts
    
    pts = _project_to_sphere(result.x.reshape(-1, 3))
    
    # Second pass: fine optimization
    result2 = minimize(
        _sphere_fun,
        pts.ravel(),
        jac=_sphere_grad,
        method='L-BFGS-B',
        options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
    )
    
    if time.time() - start_time > time_budget:
        return pts
    
    pts = _project_to_sphere(result2.x.reshape(-1, 3))
    return pts


def _random_points(rng):
    """Generate random points on the unit sphere."""
    pts = rng.standard_normal((N, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def construct():
    """Main function: optimize logarithmic energy of 100 points on sphere."""
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s safety margin
    
    rng = np.random.default_rng(42)
    
    # Start with Fibonacci spiral
    best_pts = _fibonacci_spiral()
    best_energy = _energy_fsum(best_pts)
    
    # Optimize the initial configuration
    pts_opt = _optimize_single(best_pts, 30.0)
    energy_opt = _energy_fsum(pts_opt)
    if energy_opt < best_energy:
        best_pts = pts_opt
        best_energy = energy_opt
    
    # Try multiple random restarts if time permits
    n_restarts = 0
    while time.time() < deadline - 20.0:
        pts_init = _random_points(rng)
        pts_opt = _optimize_single(pts_init, 15.0)
        energy_opt = _energy_fsum(pts_opt)
        n_restarts += 1
        
        if energy_opt < best_energy:
            best_pts = pts_opt
            best_energy = energy_opt
    
    # Final polish with tighter tolerance if time permits
    if time.time() < deadline - 5.0:
        result_final = minimize(
            _sphere_fun,
            best_pts.ravel(),
            jac=_sphere_grad,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13}
        )
        pts_final = _project_to_sphere(result_final.x.reshape(-1, 3))
        energy_final = _energy_fsum(pts_final)
        
        if energy_final < best_energy:
            best_pts = pts_final
            best_energy = energy_final
    
    return best_pts
# EVOLVE-BLOCK-END
