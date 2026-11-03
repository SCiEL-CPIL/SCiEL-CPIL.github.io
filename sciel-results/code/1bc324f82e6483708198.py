"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def _energy_and_grad(pts):
    """Compute log energy and gradient for points on unit sphere.
    Energy = sum_{i<j} log(1/|x_i - x_j|)
    Gradient with respect to x_i (Euclidean), then projected to sphere.
    """
    n = len(pts)
    # Pairwise distances
    # For N=150, we can compute all pairs
    dists = np.zeros((n, n))
    for i in range(n):
        diff = pts[i] - pts[i+1:]
        dists[i, i+1:] = np.linalg.norm(diff, axis=1)
    dists = np.maximum(dists, 1e-12)  # avoid log(0)
    
    # Energy
    energy = 0.0
    for i in range(n):
        energy += np.sum(np.log(1.0 / dists[i, i+1:]))
    
    # Gradient: d/dx_i log(1/|x_i - x_j|) = -(x_i - x_j)/|x_i - x_j|^2
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(i + 1, n):
            d = dists[i, j]
            diff = pts[i] - pts[j]
            g = -diff / (d * d)
            grad[i] += g
            grad[j] -= g
    
    # Project gradient to tangent space of sphere
    for i in range(n):
        grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
    
    return energy, grad


def _fibonacci_sphere(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return pts / norms


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Best solution found so far
    best_pts = _fibonacci_sphere(N)
    best_energy = _energy_and_grad(best_pts)[0]
    
    # Multiple restarts with different initializations
    rng = np.random.RandomState(42)
    
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        
        # Choose initial point
        if restarts == 1:
            pts = _fibonacci_sphere(N).copy()
        elif restarts == 2:
            pts = _fibonacci_sphere(N).copy()
            # Add small perturbation
            pts = _project_to_sphere(pts + 0.01 * rng.randn(N, 3))
        elif restarts == 3:
            # Random initialization
            pts = rng.randn(N, 3)
            pts = _project_to_sphere(pts)
        elif restarts == 4:
            # Fibonacci with larger perturbation
            pts = _fibonacci_sphere(N).copy()
            pts = _project_to_sphere(pts + 0.05 * rng.randn(N, 3))
        else:
            # More random with varying scales
            scale = 0.01 * (1 + 0.1 * restarts)
            pts = _fibonacci_sphere(N).copy()
            pts = _project_to_sphere(pts + scale * rng.randn(N, 3))
        
        # Optimize using L-BFGS-B
        def objective_and_grad(x):
            if time.time() > deadline - 5.0:
                # Return cached best if we're running out of time
                return best_energy, np.zeros_like(x)
            p = _project_to_sphere(x.reshape(N, 3))
            e, g = _energy_and_grad(p)
            # Gradient for flat parameterization: project back
            g_flat = (g - np.einsum('ij,ij->i', g, p)[:, None] * p).flatten()
            return e, g_flat
        
        # Try optimization
        try:
            result = minimize(
                objective_and_grad,
                pts.flatten(),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1000}
            )
            
            if result.success or result.fun < best_energy:
                new_pts = _project_to_sphere(result.x.reshape(N, 3))
                new_energy = _energy_and_grad(new_pts)[0]
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pts = new_pts.copy()
                    
        except Exception:
            pass
        
        # Check if we have time for another restart
        if time.time() > deadline - 10.0:
            break
        
        # If we've converged well, do a final high-precision polish
        if best_energy < -2364.0 and time.time() < deadline - 20.0:
            # Polish with tighter tolerances
            try:
                def polish_obj(x):
                    p = _project_to_sphere(x.reshape(N, 3))
                    e, g = _energy_and_grad(p)
                    g_flat = (g - np.einsum('ij,ij->i', g, p)[:, None] * p).flatten()
                    return e, g_flat
                
                result2 = minimize(
                    polish_obj,
                    best_pts.flatten(),
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 5000}
                )
                
                if result2.fun < best_energy:
                    new_pts = _project_to_sphere(result2.x.reshape(N, 3))
                    new_energy = _energy_and_grad(new_pts)[0]
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_pts = new_pts.copy()
                        
            except Exception:
                pass
            break
    
    # Final validation
    best_pts = _project_to_sphere(best_pts)
    
    # Check for very close points
    min_dist = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 1e-9:
        # Fallback to Fibonacci
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
# EVOLVE-BLOCK-END
