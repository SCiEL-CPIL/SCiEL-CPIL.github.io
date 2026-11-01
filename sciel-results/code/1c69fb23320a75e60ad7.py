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
    E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    d_ij = ||p_i - p_j||
    
    Gradient with respect to p_i:
    grad_i = sum_{j != i} (p_i - p_j) / d_ij^2
    (Note: The derivative of -log(d) is -1/d * d/dp_i (d) = -1/d * (p_i - p_j)/d = -(p_i - p_j)/d^2)
    Wait, let's re-verify.
    E = - sum_{i<j} ln(||p_i - p_j||)
    dE/dp_k = - sum_{j != k} (1/||p_k - p_j||) * (1/||p_k - p_j||) * (p_k - p_j)
            = - sum_{j != k} (p_k - p_j) / ||p_k - p_j||^2
    
    This is the gradient in Euclidean space.
    To project onto the tangent space of the sphere (since points must remain on the sphere),
    we project the gradient: grad_tangent = grad - (grad . p_k) * p_k
    Since p_k is unit length, p_k . p_k = 1.
    """
    if not np.all(np.isfinite(points)):
        return float('inf'), np.zeros_like(points)
    
    # Compute pairwise distances
    # dists[i, j] = ||p_i - p_j||
    dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=-1)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dists, 1.0)
    
    # Check for invalid close points
    min_dist = np.min(dists)
    if min_dist < 1e-9:
        return float('inf'), np.zeros_like(points)
        
    # Energy
    # Only sum upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    d = dists[iu]
    energy = -np.sum(np.log(d))
    
    # Gradient
    # grad[i] = - sum_{j != i} (p_i - p_j) / d_ij^2
    # We can compute this using matrix operations or loops.
    # Let's use a loop for clarity and safety, N=100 is small.
    grad = np.zeros_like(points)
    for i in range(N):
        # Vectorized over j for fixed i
        # d_ij is dists[i, :]
        # p_i - p_j is points[i] - points
        # term = (points[i] - points) / (dists[i, :]**2)[:, np.newaxis]
        # But we need to exclude j=i. dists[i,i] is 1.0, so term[i] is 0.
        # However, the formula is valid for j != i.
        # Let's compute the sum for all j, then subtract the i-th term (which is 0 anyway if we mask).
        
        # Actually, simpler:
        # diff = points[i] - points  # shape (N, 3)
        # dist_sq = dists[i, :]**2   # shape (N,)
        # term = diff / dist_sq[:, np.newaxis]
        # term[i] = 0
        # grad[i] = -np.sum(term, axis=0)
        
        diff = points[i] - points
        dist_sq = dists[i, :]**2
        # Avoid division by zero just in case, though diagonal is 1.
        dist_sq[i] = 1.0 
        term = diff / dist_sq[:, np.newaxis]
        term[i] = 0.0
        grad[i] = -np.sum(term, axis=0)
        
    # Project gradient onto tangent space
    # grad_tangent[i] = grad[i] - (grad[i] . points[i]) * points[i]
    for i in range(N):
        dot = np.dot(grad[i], points[i])
        grad[i] -= dot * points[i]
        
    return energy, grad

def _energy(points):
    """Calculate the logarithmic energy. Returns inf if invalid."""
    e, _ = _energy_and_grad(points)
    return e

def _fibonacci_sphere(n):
    """Generate points on a sphere using the Fibonacci lattice."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n):
    """Generate random points on the sphere."""
    pts = np.random.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1)
    return pts / norms[:, np.newaxis]

def _optimize(points, deadline):
    """Run L-BFGS-B optimization on the flattened points."""
    x0 = points.flatten()
    
    def objective(x):
        pts = x.reshape(N, 3)
        # Normalize to ensure constraint satisfaction during optimization steps
        # L-BFGS-B doesn't handle equality constraints well, so we project back
        norms = np.linalg.norm(pts, axis=1)
        if np.any(norms < 1e-9):
            return float('inf')
        pts = pts / norms[:, np.newaxis]
        
        # Check time
        if time.time() > deadline:
            return float('inf')
            
        e, g = _energy_and_grad(pts)
        if not np.isfinite(e):
            return float('inf')
            
        # Project gradient back to original dimensionality? 
        # The gradient returned by _energy_and_grad is already tangent.
        # But L-BFGS-B expects gradient in the space of x (Euclidean).
        # Since we are projecting x to sphere inside objective, the effective gradient
        # in Euclidean space for the constrained problem is the tangent gradient.
        # However, standard practice for manifold optimization with Euclidean optimizers
        # is to just use the Euclidean gradient of the extension of the function.
        # Here, the function is defined on R^3N but we project.
        # A safer approach: use the Euclidean gradient of the energy function 
        # defined on R^3N (which is what _energy_and_grad computes before projection? 
        # No, _energy_and_grad computes the gradient on the sphere).
        
        # Let's re-evaluate.
        # If we define f(x) = E(proj(x)), then grad f(x) = J_proj^T * grad_E(proj(x)).
        # For small steps, this is complex.
        # Alternative: Use a custom step or a different optimizer.
        # But L-BFGS-B is robust.
        # Let's try providing the Euclidean gradient of the energy function 
        # evaluated at the projected points.
        # The energy function E(p) is defined for p on sphere.
        # If we extend E to R^3N by E(x) = E(x/||x||), 
        # then grad E(x) = (1/||x||) * P_x^T * grad_E_sphere(x/||x||) ?
        # Actually, simpler: just use the gradient computed on the sphere.
        # Many implementations of sphere-constrained optimization just use the 
        # tangent gradient and let the optimizer handle the "constraint" via 
        # the objective's behavior or by re-projecting.
        # However, L-BFGS-B minimizes in Euclidean space.
        # If we give it the tangent gradient, it will move in that direction in Euclidean space.
        # Then we project. This is a standard heuristic (Riemannian gradient descent 
        # with Euclidean line search).
        
        # Let's return the tangent gradient as the Euclidean gradient.
        # This is effectively doing:
        # x_new = x - alpha * grad_tangent
        # x_proj = x_new / ||x_new||
        # This is a valid Riemannian step.
        
        return e, g.flatten()
        
    # Use L-BFGS-B
    # maxiter needs to be large enough
    res = minimize(objective, x0, method='L-BFGS-B', jac=True, 
                   options={'maxiter': 500, 'maxfun': 1000, 'gtol': 1e-8})
    
    if res.success or np.isfinite(res.fun):
        best_pts = res.x.reshape(N, 3)
        # Final normalization
        norms = np.linalg.norm(best_pts, axis=1)
        if np.all(norms > 1e-9):
            best_pts = best_pts / norms[:, np.newaxis]
            return best_pts, res.fun
    return points, float('inf')

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Multiple initializations
    inits = []
    
    # 1. Fibonacci sphere
    inits.append(_fibonacci_sphere(N))
    
    # 2. Random points (several)
    for _ in range(5):
        inits.append(_random_sphere(N))
        
    # 3. Perturbed Fibonacci
    fib = _fibonacci_sphere(N)
    noise = np.random.randn(N, 3) * 0.1
    perturbed = fib + noise
    norms = np.linalg.norm(perturbed, axis=1)
    perturbed = perturbed / norms[:, np.newaxis]
    inits.append(perturbed)
    
    best_points = None
    best_energy = float('inf')
    
    for i, init in enumerate(inits):
        # Check time
        if time.time() > deadline - 10.0: # Leave 10s for final checks
            break
            
        # Optimize
        opt_points, energy = _optimize(init, deadline)
        
        if energy < best_energy:
            best_energy = energy
            best_points = opt_points.copy()
            
        # Print progress for debugging (optional)
        # print(f"Init {i}: Energy {energy:.6f}")
        
    if best_points is None:
        return _fibonacci_sphere(N)
        
    return best_points
# EVOLVE-BLOCK-END
