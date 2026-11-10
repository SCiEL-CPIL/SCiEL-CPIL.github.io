"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _energy_and_grad(pts):
    """
    Compute the logarithmic energy and its gradient for points on the sphere.
    Energy E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    Gradient for point i: grad_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    
    We compute the Euclidean gradient and then project it to the tangent space 
    in the optimization wrapper, or we can compute the Riemannian gradient directly.
    Here we return the Euclidean gradient, and the wrapper will project it.
    """
    n = len(pts)
    # Pairwise differences: (n, n, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    # Pairwise distances: (n, n)
    dist = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero on the diagonal
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum over i < j
    triu_indices = np.triu_indices(n, k=1)
    dist_triu = dist[triu_indices]
    energy = -np.sum(np.log(dist_triu))
    
    # Gradient calculation
    # grad_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # Let W_ij = 1 / |x_i - x_j|^2 for i != j, and 0 for i == j
    inv_dist_sq = 1.0 / (dist * dist)
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[i] = -sum_j diff[i, j] * inv_dist_sq[i, j]
    # diff[i, j] is (x_i - x_j)
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad

def _project_to_tangent(pts, grad):
    """
    Project the Euclidean gradient to the tangent space of the unit sphere.
    T_p S^2 = { v in R^3 | v . p = 0 }
    Projection: g_tan = g - (g . p) p
    """
    # pts: (n, 3), grad: (n, 3)
    # Dot product: (n,)
    dot = np.sum(pts * grad, axis=1, keepdims=True)
    grad_tan = grad - dot * pts
    return grad_tan

def _normalize(pts):
    """Normalize points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _make_fibonacci_sphere(n):
    """Generate points on the sphere using the Fibonacci spiral method."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z)) # Clamp to avoid sqrt of negative due to FP
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _random_sphere(n, seed):
    """Generate random points on the sphere."""
    rng = np.random.default_rng(seed)
    # Generate random vectors and normalize
    pts = rng.normal(size=(n, 3))
    return _normalize(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    best_pts = None
    best_energy = float('inf')
    
    # List of initial configurations
    initializations = []
    
    # 1. Fibonacci spiral
    initializations.append(_make_fibonacci_sphere(N))
    
    # 2. Random points (several seeds)
    for seed in [1, 2, 3, 42, 123]:
        initializations.append(_random_sphere(N, seed))
        
    # 3. Perturbed Fibonacci
    rng = np.random.default_rng(99)
    fib = _make_fibonacci_sphere(N)
    for scale in [0.05, 0.1, 0.2]:
        perturbed = fib + rng.normal(0, scale, size=fib.shape)
        initializations.append(_normalize(perturbed))

    # Optimization function wrapper
    def obj_grad(x_flat):
        pts = x_flat.reshape(N, 3)
        # Normalize to ensure we stay on the manifold during intermediate steps
        # Note: L-BFGS-B steps in R^3, so we must normalize before calculating energy/grad
        # However, the gradient of the normalized energy is not simply the projected gradient of the unnormalized energy.
        # Standard approach: minimize f(p) where p is on sphere.
        # If we use L-BFGS-B on R^3, we are effectively doing:
        # p_new = p - alpha * g_proj
        # Then normalize p_new.
        # This is a heuristic. A more rigorous method is Riemannian optimization.
        # Given the time constraint, this heuristic usually works well for this problem.
        
        e, g = _energy_and_grad(pts)
        g_tan = _project_to_tangent(pts, g)
        return e, g_tan.ravel()

    # Run optimization for each initialization
    for i, init_pts in enumerate(initializations):
        if time.time() > deadline - 10.0:
            break
            
        current_energy, _ = _energy_and_grad(init_pts)
        
        # If this initialization is already worse than the best found, skip if we have enough restarts
        if current_energy > best_energy and i > 2:
             # Allow a small tolerance for exploration
             if current_energy - best_energy > 5.0:
                 continue

        x0 = init_pts.ravel()
        
        try:
            # L-BFGS-B with tight tolerances
            result = minimize(
                obj_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 1000,
                    'ftol': 1e-14,
                    'gtol': 1e-10,
                    'maxls': 50,
                    'disp': False
                }
            )
            
            # Re-normalize the result
            res_pts = _normalize(result.x.reshape(N, 3))
            
            # Calculate final energy accurately
            res_energy, _ = _energy_and_grad(res_pts)
            
            if res_energy < best_energy:
                best_energy = res_energy
                best_pts = res_pts.copy()
                
        except Exception as e:
            # If optimization fails, keep the initial points if they are better
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = init_pts.copy()
                
    # If we still have time, do a final polish on the best configuration
    if best_pts is not None and time.time() < deadline - 5.0:
        try:
            x0 = best_pts.ravel()
            result = minimize(
                obj_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 2000,
                    'ftol': 1e-15,
                    'gtol': 1e-12,
                    'maxls': 100,
                    'disp': False
                }
            )
            res_pts = _normalize(result.x.reshape(N, 3))
            res_energy, _ = _energy_and_grad(res_pts)
            
            if res_energy < best_energy:
                best_energy = res_energy
                best_pts = res_pts.copy()
        except:
            pass

    # Final validity checks
    if best_pts is None:
        best_pts = _make_fibonacci_sphere(N)
        
    # Ensure no points are too close
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        # Fallback to Fibonacci if invalid
        best_pts = _make_fibonacci_sphere(N)
        
    if not np.all(np.isfinite(best_pts)):
        best_pts = _make_fibonacci_sphere(N)

    return best_pts

# EVOLVE-BLOCK-END
