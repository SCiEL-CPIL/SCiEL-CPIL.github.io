"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin before 150s kill

    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    x0 = x0 / np.linalg.norm(x0, axis=1, keepdims=True)

    best_x = x0.copy()
    best_energy = _energy(x0)

    def _energy(x):
        # Compute pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dist, 1.0)  # avoid log(0)
        # Only sum i < j
        iu = np.triu_indices(N, 1)
        d = dist[iu]
        return math.fsum(math.log(1.0 / d) for d in d)

    def _grad(x):
        # Gradient of sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log(|xi-xj|)
        # d/dxi log(1/|xi-xj|) = - (xi - xj) / |xi-xj|^2
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist2 = np.sum(diff * diff, axis=-1)
        dist = np.sqrt(dist2)
        np.fill_diagonal(dist2, 1.0)
        np.fill_diagonal(dist, 1.0)
        
        # For each pair i<j, gradient contribution to i is -(xi-xj)/d^2, to j is (xi-xj)/d^2
        # So grad_i = -sum_{j!=i} (xi - xj) / |xi-xj|^2
        # = sum_{j!=i} (xj - xi) / |xi-xj|^2
        inv_dist2 = 1.0 / dist2
        np.fill_diagonal(inv_dist2, 0.0)
        
        # grad = sum over j of (xj - xi) / |xi-xj|^2
        # = (sum_j xj * inv_dist2) - xi * sum_j inv_dist2
        sum_xj_inv = np.sum(x * inv_dist2, axis=1)
        sum_inv = np.sum(inv_dist2, axis=1)
        grad = sum_xj_inv - x * sum_inv[:, np.newaxis]
        return grad

    def _project_to_sphere(x):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms < 1e-15, 1.0, norms)
        return x / norms

    def _energy_and_grad(x):
        x = _project_to_sphere(x)
        e = _energy(x)
        g = _grad(x)
        # Project gradient to tangent space
        g = g - np.sum(g * x, axis=1, keepdims=True) * x
        return e, g

    # Use L-BFGS-B with projection via re-parameterization
    # We'll use scipy's minimize with SLSQP or L-BFGS-B on the sphere
    # Better: use a custom approach with gradient descent + line search

    # Try scipy minimize with bounds and constraints
    # Actually, let's use a simpler approach: iterative gradient descent with backtracking line search
    
    x = x0.copy()
    lr = 0.01
    max_iter = 50000
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        e, g = _energy_and_grad(x)
        
        # Update best
        if e < best_energy:
            best_energy = e
            best_x = x.copy()
        
        # Backtracking line search
        step = lr
        x_new = x - step * g
        x_new = _project_to_sphere(x_new)
        
        e_new = _energy(x_new)
        
        # Armijo condition
        c1 = 1e-4
        while e_new > e + c1 * step * np.sum(g * (x_new - x)):
            step *= 0.5
            x_new = x - step * g
            x_new = _project_to_sphere(x_new)
            e_new = _energy(x_new)
            if step < 1e-12:
                break
        
        x = x_new
        lr = step * 1.05  # slightly increase step size if we took a full step
        
        # Convergence check
        if it > 100 and abs(e_new - e) < 1e-10:
            break

    # Final polish with higher precision
    if time.time() < deadline - 5.0:
        x = best_x.copy()
        for it in range(10000):
            if time.time() > deadline:
                break
            e, g = _energy_and_grad(x)
            if e < best_energy:
                best_energy = e
                best_x = x.copy()
            
            step = 0.01
            x_new = x - step * g
            x_new = _project_to_sphere(x_new)
            e_new = _energy(x_new)
            
            c1 = 1e-4
            while e_new > e + c1 * step * np.sum(g * (x_new - x)):
                step *= 0.5
                x_new = x - step * g
                x_new = _project_to_sphere(x_new)
                e_new = _energy(x_new)
                if step < 1e-14:
                    break
            
            x = x_new
            if step < 1e-12:
                break

    # Ensure all points are on the unit sphere
    best_x = _project_to_sphere(best_x)
    
    # Validate
    if not np.all(np.isfinite(best_x)):
        return x0
    if np.any(np.linalg.norm(best_x, axis=1) < 1e-15):
        return x0
    
    return best_x
# EVOLVE-BLOCK-END
