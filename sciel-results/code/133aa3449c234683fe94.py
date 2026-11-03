"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _log_energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on unit sphere.
    
    Energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient w.r.t. x_i: sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist = np.linalg.norm(diff, axis=2)  # (n, n)
    
    # Avoid division by zero
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum_{i<j} log(1/dist_ij) = -sum_{i<j} log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each i, grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # diff[i,j] = x_i - x_j
    # grad_i = sum_j diff[i,j] / dist[i,j]^2
    inv_dist2 = np.zeros_like(dist)
    np.fill_diagonal(inv_dist2, 0.0)
    inv_dist2[dist > 1e-12] = 1.0 / (dist[dist > 1e-12] ** 2)
    
    # grad[i] = sum_j diff[i,j] * inv_dist2[i,j]
    grad = np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _optimize_from_start(pts0, time_budget):
    """Run L-BFGS optimization from initial points, with time budget."""
    start_time = time.time()
    
    def objective_and_grad(x_flat):
        pts = x_flat.reshape(-1, 3)
        e, g = _log_energy_and_grad(pts)
        # Project gradient to tangent space: remove radial component
        # For each point i, radial direction is pts[i]/|pts[i]|
        # Since we're on sphere, |pts[i]| = 1, so radial = pts[i]
        # Tangent grad = g_i - (g_i . pts[i]) * pts[i]
        radial_comp = np.sum(g * pts, axis=1, keepdims=True)
        g_tangent = g - radial_comp * pts
        return e, g_tangent.flatten()
    
    x0 = pts0.flatten()
    
    best_x = x0.copy()
    best_e = objective_and_grad(x0)[0]
    
    # Use L-BFGS with custom objective
    for _ in range(50):  # max 50 outer iterations (each may do multiple line searches)
        if time.time() - start_time > time_budget:
            break
        
        res = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'gtol': 1e-10, 'ftol': 1e-12}
        )
        
        x_new = res.x
        # Project back to sphere
        pts_new = _project_to_sphere(x_new.reshape(-1, 3))
        e_new = _log_energy_and_grad(pts_new)[0]
        
        if e_new < best_e:
            best_e = e_new
            best_x = pts_new.flatten()
        
        # If convergence is good, break
        if res.success and res.status == 0:
            break
        
        # Update x0 for next iteration (helps with numerical issues)
        x0 = pts_new.flatten()
    
    return _project_to_sphere(best_x.reshape(-1, 3)), best_e


def _fibonacci_sphere(n, offset=0.5):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    """Generate random points on unit sphere."""
    v = rng.standard_normal((n, 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms


def construct():
    """Construct 100 points on unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    deadline = 115.0  # safety margin
    
    rng = np.random.RandomState(42)
    
    best_pts = None
    best_energy = float('inf')
    
    # Strategy 1: Fibonacci spiral start
    pts0 = _fibonacci_sphere(N)
    pts, e = _optimize_from_start(pts0, 30.0)
    if e < best_energy:
        best_energy = e
        best_pts = pts
    
    # Strategy 2: Random restarts
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() - start_time > deadline - 5.0:
            break
        
        # Mix of random and perturbed Fibonacci
        if r % 2 == 0:
            pts0 = _random_sphere(N, rng)
        else:
            # Perturbed Fibonacci
            pts0 = _fibonacci_sphere(N) + rng.standard_normal((N, 3)) * 0.1
            pts0 = _project_to_sphere(pts0)
        
        budget = min(10.0, deadline - (time.time() - start_time))
        if budget < 1.0:
            break
        
        pts, e = _optimize_from_start(pts0, budget)
        if e < best_energy:
            best_energy = e
            best_pts = pts
    
    # Final polish: one more optimization from best point with tighter tolerance
    if time.time() - start_time < deadline - 5.0 and best_pts is not None:
        pts, e = _optimize_from_start(best_pts, deadline - (time.time() - start_time) - 2.0)
        if e < best_energy:
            best_energy = e
            best_pts = pts
    
    # Ensure all points are on unit sphere and valid
    best_pts = _project_to_sphere(best_pts)
    
    # Validate: no points too close
    min_dist = float('inf')
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 1e-9:
        # Fallback to Fibonacci
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
# EVOLVE-BLOCK-END
