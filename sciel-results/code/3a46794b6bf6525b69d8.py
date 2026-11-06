"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


def _log_energy(points):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    # points: (N, 3) array
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Mask diagonal and upper triangle
    iu = np.triu_indices(N, 1)
    d = dist[iu]
    # Avoid log(0) or negative values
    d = np.maximum(d, 1e-15)
    return math.fsum(math.log(1.0 / di) for di in d)


def _log_energy_grad(points):
    """Gradient of logarithmic energy w.r.t. points on sphere (projected)."""
    # points: (N, 3)
    # grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2 * 2
    # Actually d/dx_i [log(1/|x_i - x_j|)] = -d/dx_i [log|x_i - x_j|] = -(x_i - x_j)/|x_i - x_j|^2
    # So gradient of energy = sum_{j != i} -(x_i - x_j)/|x_i - x_j|^2
    # But we want to minimize, so gradient points in direction of increase.
    
    grad = np.zeros_like(points)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            diff = points[i] - points[j]
            d2 = np.dot(diff, diff)
            if d2 < 1e-20:
                continue
            grad[i] += -diff / d2
    
    # Project gradient to tangent space of sphere
    # For each point, subtract component along the point itself
    for i in range(N):
        grad[i] -= np.dot(grad[i], points[i]) * points[i]
    
    return grad


def _project_to_sphere(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def construct():
    """
    Use a multi-start optimization approach:
    1. Generate multiple initial configurations (Fibonacci spiral with perturbations)
    2. For each, run L-BFGS-B optimization with sphere constraint via reparameterization
    3. Return the best configuration found
    """
    best_points = None
    best_energy = float('inf')
    
    # Use spherical coordinates parameterization: theta, phi for each point
    # Or better: use the fact that we can optimize in R^3 and project
    
    # Strategy: Use L-BFGS with gradient projection onto tangent space
    # Initialize with Fibonacci spiral + random perturbations
    
    np.random.seed(42)
    
    for trial in range(5):
        # Create initial configuration
        if trial == 0:
            # Fibonacci spiral
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(1.0 - z * z)
            x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        else:
            # Fibonacci spiral with random perturbation
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i + np.random.uniform(-0.1, 0.1, N)
            r = np.sqrt(1.0 - z * z)
            x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            # Add small random perturbation
            x0 += np.random.normal(0, 0.05, size=(N, 3))
            x0 = _project_to_sphere(x0)
        
        # Flatten for optimizer
        x_flat = x0.flatten()
        
        def objective_and_grad(x_flat):
            pts = x_flat.reshape(N, 3)
            # Project to sphere
            pts = _project_to_sphere(pts)
            e = _log_energy(pts)
            g = _log_energy_grad(pts)
            return e, g.flatten()
        
        # Use L-BFGS-B
        try:
            res = minimize(
                objective_and_grad,
                x_flat,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-10}
            )
            pts = _project_to_sphere(res.x.reshape(N, 3))
            e = _log_energy(pts)
            
            if e < best_energy:
                best_energy = e
                best_points = pts
        except Exception:
            continue
    
    # If no optimization succeeded, fall back to Fibonacci spiral
    if best_points is None:
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    return best_points
# EVOLVE-BLOCK-END
