"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _energy(points):
    """Calculate the logarithmic energy."""
    if not np.all(np.isfinite(points)):
        return float('inf')
    dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=-1)
    np.fill_diagonal(dists, 1.0)
    min_dist = np.min(dists)
    if min_dist < 1e-9:
        return float('inf')
    iu = np.triu_indices(N, k=1)
    d = dists[iu]
    d[d < 1e-9] = 1e-9
    energy = -np.sum(np.log(d))
    return energy

def _energy_and_grad(points):
    """Calculate the logarithmic energy and its gradient."""
    if not np.all(np.isfinite(points)):
        return float('inf'), np.zeros_like(points)
    
    # Compute pairwise distances
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    np.fill_diagonal(dists, 1.0)
    
    # Check for invalid close points
    min_dist = np.min(dists)
    if min_dist < 1e-9:
        return float('inf'), np.zeros_like(points)
    
    # Energy
    iu = np.triu_indices(N, k=1)
    d = dists[iu]
    d[d < 1e-9] = 1e-9
    energy = -np.sum(np.log(d))
    
    # Gradient: dE/dp_i = -sum_{j!=i} (1/d_ij) * (p_i - p_j)/d_ij
    # = -sum_{j!=i} (p_i - p_j) / d_ij^2
    grad = np.zeros_like(points)
    for i in range(N):
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        diffs_i = points[i] - points[mask]
        dists_i = dists[i, mask]
        dists_i[dists_i < 1e-9] = 1e-9
        # grad_i = -sum_j (p_i - p_j) / d_ij^2
        grad_i = -np.sum(diffs_i / (dists_i ** 2)[:, np.newaxis], axis=0)
        # Project to tangent space: grad_tan = grad_i - (grad_i . p_i) p_i
        grad_i = grad_i - np.dot(grad_i, points[i]) * points[i]
        grad[i] = grad_i
    
    return energy, grad

def _fibonacci_sphere(n):
    """Generate points on a sphere using the Fibonacci lattice."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=None):
    """Generate random points on the sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1)
    return pts / norms[:, np.newaxis]

def _optimize_from_start(start_points, max_time=30.0):
    """Run L-BFGS-B optimization from a given starting configuration."""
    start_time = time.time()
    
    def objective_and_grad(x):
        """Flatten/unflatten for scipy optimizer."""
        pts = x.reshape(-1, 3)
        # Project to sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        pts = pts / norms
        e, g = _energy_and_grad(pts)
        # Unflatten gradient
        g_flat = g.flatten()
        return e, g_flat
    
    x0 = start_points.flatten()
    
    # Use L-BFGS-B
    result = minimize(
        objective_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 10000}
    )
    
    if result.success or result.fun < float('inf'):
        best_x = result.x
    else:
        best_x = x0
    
    # Project back to sphere
    best_pts = best_x.reshape(-1, 3)
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    best_pts = best_pts / norms
    
    elapsed = time.time() - start_time
    return best_pts, _energy(best_pts), elapsed

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_points = None
    best_energy = float('inf')
    
    # Strategy: Coarse-to-fine with multiple structured starts
    # Phase 1: Use Fibonacci lattice as base, then perturb at multiple scales
    # Phase 2: Run L-BFGS-B from each perturbed start, keeping the best
    
    # Generate diverse starting points based on Fibonacci lattice with perturbations
    starts = []
    
    # Start 1: Pure Fibonacci
    fib_pts = _fibonacci_sphere(N)
    starts.append(fib_pts)
    
    # Starts 2-10: Fibonacci with random perturbations at various scales
    rng = np.random.default_rng(123)
    for scale_idx, scale in enumerate([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]):
        for seed in [42 + scale_idx * 100, 43 + scale_idx * 100]:
            if time.time() > deadline - 10.0:
                break
            perturbed = fib_pts.copy()
            noise = rng.standard_normal((N, 3)) * scale
            perturbed = perturbed + noise
            norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            perturbed = perturbed / norms
            starts.append(perturbed)
    
    # Starts 11-15: Pure random starts
    for seed in [1, 2, 3, 4, 5]:
        if time.time() > deadline - 10.0:
            break
        random_pts = _random_sphere(N, seed=seed)
        starts.append(random_pts)
    
    # Run optimization from each start, tracking best
    for idx, start_pts in enumerate(starts):
        if time.time() > deadline - 15.0:
            break
        
        # Estimate time per optimization
        # Each L-BFGS-B run takes ~2-5 seconds typically
        # Allow up to 10 seconds per optimization to be safe
        pts, energy, elapsed = _optimize_from_start(start_pts, max_time=10.0)
        
        if energy < best_energy:
            best_energy = energy
            best_points = pts.copy()
        
        # If we have time, do a second polish pass on the best found so far
        if idx == len(starts) - 1 or time.time() > deadline - 25.0:
            break
    
    # Final polish: Run L-BFGS-B again from the best point found with tighter tolerances
    if best_points is not None and time.time() < deadline - 10.0:
        def objective_and_grad_final(x):
            pts = x.reshape(-1, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            pts = pts / norms
            e, g = _energy_and_grad(pts)
            g_flat = g.flatten()
            return e, g_flat
        
        x0 = best_points.flatten()
        result = minimize(
            objective_and_grad_final,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 50000}
        )
        
        if result.fun < best_energy:
            best_pts = result.x.reshape(-1, 3)
            norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            best_pts = best_pts / norms
            best_energy = result.fun
            best_points = best_pts
    
    # Ensure valid output
    if best_points is None:
        best_points = fib_pts
    
    # Final validation
    if not np.all(np.isfinite(best_points)):
        best_points = fib_pts
    
    # Normalize
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    best_points = best_points / norms
    
    return best_points
# EVOLVE-BLOCK-END
