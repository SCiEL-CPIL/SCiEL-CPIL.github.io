"""Rewritten program using gradient-based optimization with repulsion forces."""
import numpy as np
from scipy.optimize import minimize
import math

N = 150

# EVOLVE-BLOCK-START

def _log_energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|xi-xj|)."""
    n = len(points)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-15:
                return 1e10  # penalty for coincident points
            total += math.log(1.0 / d)
    return total

def _log_energy_and_grad(points):
    """Compute logarithmic energy and its gradient.
    
    Energy E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    Gradient w.r.t. point i: -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(points)
    E = 0.0
    grad = np.zeros_like(points)
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = points[i] - points[j]
            d2 = np.dot(diff, diff)
            if d2 < 1e-15:
                return 1e10, grad
            d = math.sqrt(d2)
            E += math.log(1.0 / d)
            # Gradient contribution: -diff / d^2 for point i, +diff / d^2 for point j
            coeff = 1.0 / d2
            grad[i] -= diff * coeff
            grad[j] += diff * coeff
    
    return E, grad

def _project_to_sphere(points):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms

def _initial_guess():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts

def construct():
    """Construct 150 points on unit sphere minimizing logarithmic energy."""
    np.random.seed(42)
    
    # Start with Fibonacci spiral
    points = _initial_guess()
    
    # Multiple restarts with perturbations, keep best
    best_points = points.copy()
    best_energy = _log_energy(points)
    
    n_restarts = 3
    for restart in range(n_restarts):
        if restart == 0:
            pts = points.copy()
        else:
            # Perturb initial guess
            pts = _initial_guess()
            noise = np.random.randn(N, 3) * 0.1
            pts = pts + noise
            pts = _project_to_sphere(pts)
        
        # Optimize on the sphere using L-BFGS-B with parameterization
        # We optimize in R^{3N} and project back, using projected gradient idea
        # Simpler: use scipy minimize with constraints via Lagrange or just project
        
        def objective_and_grad(x):
            pts = _project_to_sphere(x.reshape(N, 3))
            E, grad = _log_energy_and_grad(pts)
            # Project gradient to tangent space
            grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
            return E, grad_tangent.flatten()
        
        x0 = pts.flatten()
        
        # Use L-BFGS-B
        result = minimize(
            objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        opt_pts = _project_to_sphere(result.x.reshape(N, 3))
        opt_energy = _log_energy(opt_pts)
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_points = opt_pts
    
    # Final polishing: local refinement with smaller steps
    # Use Nelder-Mead or further L-BFGS from best point
    x0 = best_points.flatten()
    
    def objective_only(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        return _log_energy(pts)
    
    result = minimize(
        objective_only,
        x0,
        method='L-BFGS-B',
        options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
    )
    
    final_points = _project_to_sphere(result.x.reshape(N, 3))
    final_energy = _log_energy(final_points)
    
    if final_energy < best_energy:
        best_points = final_points
    
    # Ensure all points are valid
    norms = np.linalg.norm(best_points, axis=1)
    for i in range(N):
        if norms[i] < 1e-15:
            best_points[i] = np.array([1.0, 0.0, 0.0])
        else:
            best_points[i] /= norms[i]
    
    # Check for near-coincident points
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                # Separate them slightly
                diff = best_points[i] - best_points[j]
                if np.linalg.norm(diff) < 1e-15:
                    diff = np.array([1.0, 0.0, 0.0])
                diff /= np.linalg.norm(diff)
                best_points[i] += diff * 1e-8
                best_points[j] -= diff * 1e-8
                best_points[i] /= np.linalg.norm(best_points[i])
                best_points[j] /= np.linalg.norm(best_points[j])
    
    return best_points

# EVOLVE-BLOCK-END
