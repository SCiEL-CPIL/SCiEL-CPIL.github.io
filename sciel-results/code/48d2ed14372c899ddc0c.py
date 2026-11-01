"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _energy_and_grad(points):
    """
    Calculate the logarithmic energy and its gradient.
    E = sum_{i<j} log(1 / |x_i - x_j|) = -sum_{i<j} log(d_ij)
    d_ij = |x_i - x_j|
    
    Gradient with respect to x_k:
    dE/dx_k = sum_{j != k} (1/d_kj) * (x_k - x_j) / |x_k - x_j|^2 
             = sum_{j != k} (x_k - x_j) / d_kj^3
    """
    # points shape (N, 3)
    # Compute pairwise distances
    # d_ij = sqrt((x_i - x_j)^2)
    # d_ij^2 = x_i^2 + x_j^2 - 2 x_i . x_j
    
    x_sq = np.sum(points * points, axis=1)
    dist_sq = x_sq[:, np.newaxis] + x_sq[np.newaxis, :] - 2.0 * (points @ points.T)
    # Clip to avoid negative zeros or small negatives due to FP
    np.maximum(dist_sq, 1e-15, out=dist_sq)
    dist = np.sqrt(dist_sq)
    
    # Mask diagonal
    np.fill_diagonal(dist, 1.0)
    np.fill_diagonal(dist_sq, 1.0)
    
    # Energy: sum over i<j of -log(dist_ij)
    # We can compute full sum and divide by 2, or use triu
    iu = np.triu_indices(N, k=1)
    d_ij = dist[iu]
    energy = -np.sum(np.log(d_ij))
    
    # Gradient
    # grad_k = sum_{j != k} (x_k - x_j) / dist_kj^3
    # Note: dist_kj^3 = dist_kj^2 * dist_kj = dist_sq_kj * dist_kj
    # Term for pair (i, j) contributes to grad_i and grad_j
    # Contribution to grad_i from j: (x_i - x_j) / dist_ij^3
    # Contribution to grad_j from i: (x_j - x_i) / dist_ij^3 = - (x_i - x_j) / dist_ij^3
    
    # Let's compute the matrix of coefficients
    # C_ij = 1 / dist_ij^3 for i != j
    # Then grad_k = sum_j C_kj * (x_k - x_j)
    
    # To avoid division by zero on diagonal, we set C_kk = 0
    C = 1.0 / (dist_sq * dist)
    np.fill_diagonal(C, 0.0)
    
    # grad_k = sum_j C_kj * x_k - sum_j C_kj * x_j
    # sum_j C_kj * x_k = x_k * sum_j C_kj
    # sum_j C_kj * x_j = (C @ points)_k
    
    row_sums = np.sum(C, axis=1)
    grad = row_sums[:, np.newaxis] * points - (C @ points)
    
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
    if seed is not None:
        np.random.seed(seed)
    v = np.random.normal(size=(n, 3))
    norms = np.linalg.norm(v, axis=1)
    return v / norms[:, np.newaxis]

def _optimize(points_init, time_limit):
    """
    Optimize points on the sphere using L-BFGS-B in Cartesian coordinates,
    projecting back to the sphere after each step to maintain the constraint.
    Note: Standard L-BFGS-B doesn't handle manifold constraints directly.
    We use a penalty method or simply project. 
    Actually, a better approach for the sphere is to use the fact that the gradient
    on the sphere is the projection of the Euclidean gradient onto the tangent space.
    grad_sphere = grad_euclidean - (grad_euclidean . x_k) * x_k
    
    However, scipy's L-BFGS-B expects unconstrained optimization.
    We can use the parameterization of the sphere or just project.
    Projection is simpler but might be less efficient.
    Let's try a simple gradient descent with line search on the sphere first, 
    or use L-BFGS-B with a penalty term for deviation from the sphere?
    
    A common trick: optimize in R^3 and add a penalty lambda * sum(||x_i||^2 - 1)^2.
    But this requires tuning lambda.
    
    Alternative: Use the fact that the problem is scale-invariant in the energy? No, log(1/d) is not scale invariant if d changes.
    
    Let's stick to a robust method: 
    Use L-BFGS-B on the tangent space? No, L-BFGS-B is for R^n.
    
    Let's use a simple projected gradient method with backtracking line search.
    It is robust and easy to implement.
    """
    current = points_init.copy()
    current_energy, _ = _energy_and_grad(current)
    
    step_size = 0.1
    max_iter = 500
    
    for _ in range(max_iter):
        if time.time() > time_limit:
            break
            
        energy, grad = _energy_and_grad(current)
        
        # Project gradient to tangent space
        # grad_tangent = grad - (grad . x) x
        dot = np.sum(grad * current, axis=1)
        grad_tangent = grad - dot[:, np.newaxis] * current
        
        # Check convergence
        if np.linalg.norm(grad_tangent) < 1e-8:
            break
        
        # Line search
        alpha = step_size
        for _ in range(20):
            new_points = current + alpha * grad_tangent
            # Normalize
            norms = np.linalg.norm(new_points, axis=1)
            new_points = new_points / norms[:, np.newaxis]
            
            new_energy, _ = _energy_and_grad(new_points)
            
            if new_energy < energy - 1e-4 * alpha * np.sum(grad_tangent * grad_tangent):
                current = new_points
                current_energy = new_energy
                break
            alpha *= 0.5
        else:
            # If line search fails, reduce step size globally
            step_size *= 0.5
            if step_size < 1e-8:
                break
            
    return current, current_energy

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_points = _fibonacci_sphere(N)
    best_energy, _ = _energy_and_grad(best_points)
    
    # Multi-start
    starts = [best_points]
    for i in range(10):
        # Random starts
        starts.append(_random_sphere(N, seed=i))
        
    for start in starts:
        if time.time() > deadline:
            break
            
        # Allocate remaining time for this optimization
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
            
        # We don't know exactly how long it will take, so we just run it
        # The _optimize function has a time_limit parameter, but it's a hard stop.
        # Better to let it run and check time in the main loop?
        # The _optimize function currently doesn't check time in the inner loop effectively
        # because it's a simple loop. Let's pass the deadline.
        
        optimized, energy = _optimize(start, deadline)
        
        if energy < best_energy:
            best_energy = energy
            best_points = optimized.copy()
            
    return best_points
# EVOLVE-BLOCK-END
