"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 150

# EVOLVE-BLOCK-START

def _energy(points):
    """Compute logarithmic energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)."""
    # Compute all pairwise distances
    # points shape (N, 3)
    # Use broadcasting for efficiency
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    # Mask diagonal (distance 0)
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    # Filter out any zero distances (shouldn't happen for valid points)
    pair_dists = pair_dists[pair_dists > 1e-15]
    return -math.fsum(np.log(pair_dists).tolist())


def _gradient(points):
    """Compute gradient of the logarithmic energy w.r.t. points on the sphere.
    
    Energy E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    dE/dx_i = -sum_{j != i} d/dx_i log(d_ij) = -sum_{j != i} (x_i - x_j) / d_ij^2
    But we need to project to the tangent space of the sphere.
    
    Actually, for constrained optimization on the sphere, we use the Riemannian gradient.
    The Euclidean gradient is:
    grad_E(x_i) = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    
    Wait, let me re-derive. E = sum_{i<j} -log(||x_i - x_j||)
    dE/dx_i = sum_{j != i} -1/||x_i - x_j|| * (x_i - x_j)/||x_i - x_j||
            = -sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    
    The Riemannian gradient is the projection of the Euclidean gradient onto the tangent space:
    grad_R(x_i) = grad_E(x_i) - <grad_E(x_i), x_i> x_i
    """
    n = len(points)
    grad = np.zeros_like(points)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = points[i] - points[j]
            dist_sq = np.dot(d, d)
            if dist_sq < 1e-15:
                continue
            grad[i] -= d / dist_sq
    
    # Project to tangent space
    for i in range(n):
        grad[i] -= np.dot(grad[i], points[i]) * points[i]
    
    return grad


def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms


def _fibonacci_spiral(n):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(n, rng):
    """Generate random points on the unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _lbfgs_minimize(points, max_iter=5000, tol=1e-12):
    """Minimize the logarithmic energy using L-BFGS-B with sphere constraint via parameterization.
    
    We use a simple approach: iterate gradient descent with projection, or use scipy.
    """
    from scipy.optimize import minimize
    
    n = len(points)
    best_energy = _energy(points)
    best_points = points.copy()
    
    # Use a simple projected gradient descent with adaptive step size
    # This avoids the complexity of scipy's constrained optimization
    
    step_size = 0.01
    for it in range(max_iter):
        grad = _gradient(points)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < tol:
            break
        
        # Adaptive step size
        new_points = points - step_size * grad
        new_points = _project_to_sphere(new_points)
        
        new_energy = _energy(new_points)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_points = new_points.copy()
            # Increase step size slightly
            step_size = min(step_size * 1.05, 0.1)
        else:
            # Decrease step size
            step_size *= 0.7
            if step_size < 1e-8:
                break
        
        points = new_points
    
    return best_points, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 10s before 150s limit
    
    best_points = None
    best_energy = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Multiple restarts with different initializations
    n_restarts = 10
    max_iters_per_restart = 3000
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        if restart == 0:
            # Start with Fibonacci spiral (known good configuration)
            points = _fibonacci_spiral(N)
        elif restart == 1:
            # Slightly perturbed Fibonacci spiral
            points = _fibonacci_spiral(N)
            points += 0.01 * rng.standard_normal((N, 3))
            points = _project_to_sphere(points)
        else:
            # Random start with some structure
            # Mix of random and perturbed Fibonacci
            if restart % 2 == 0:
                points = _fibonacci_spiral(N)
                points += 0.05 * rng.standard_normal((N, 3))
            else:
                points = _random_points(N, rng)
            points = _project_to_sphere(points)
        
        # Optimize
        opt_points, opt_energy = _lbfgs_minimize(points, max_iter=max_iters_per_restart)
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_points = opt_points.copy()
        
        # Check time
        if time.time() > deadline - 10.0:
            break
    
    # Final high-precision polish
    if best_points is not None and time.time() < deadline - 2.0:
        polished, polished_energy = _lbfgs_minimize(best_points, max_iter=10000, tol=1e-14)
        if polished_energy < best_energy:
            best_points = polished
            best_energy = polished_energy
    
    # Validate
    if best_points is None:
        # Fallback to Fibonacci spiral
        best_points = _fibonacci_spiral(N)
    
    # Ensure all points are on the unit sphere and finite
    best_points = _project_to_sphere(best_points)
    if not np.all(np.isfinite(best_points)):
        best_points = _fibonacci_spiral(N)
    
    return best_points

# EVOLVE-BLOCK-END
