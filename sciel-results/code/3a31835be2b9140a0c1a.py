"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _energy_and_grad(x):
    """
    Compute logarithmic energy and gradient for N points on unit sphere.
    x: (N, 3) array of points on unit sphere.
    Returns (energy, grad) where grad is (N, 3).
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient of energy w.r.t. x_i: -sum_{j!=i} (x_i - x_j)/|x_i - x_j|^2 * (-1)
    Wait, let's be careful.
    E = sum_{i<j} -log(d_{ij}) where d_{ij} = |x_i - x_j|
    dE/dx_i = sum_{j != i} - (1/d_{ij}) * d(d_{ij})/dx_i
    d(d_{ij})/dx_i = (x_i - x_j)/d_{ij}
    So dE/dx_i = sum_{j != i} - (1/d_{ij}) * (x_i - x_j)/d_{ij} = -sum_{j != i} (x_i - x_j)/d_{ij}^2
    """
    # Compute pairwise distances
    # x: (N, 3)
    # diff: (N, N, 3)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]  # diff[i,j] = x_i - x_j
    dist = np.linalg.norm(diff, axis=2)  # (N, N)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist, 1.0)  # Temporary, will mask out
    
    # Energy: sum over i<j of -log(dist[i,j])
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient
    # grad[i] = -sum_{j != i} (x_i - x_j) / dist[i,j]^2
    # We can compute this efficiently
    # inv_dist_sq: (N, N) where inv_dist_sq[i,j] = 1/dist[i,j]^2
    inv_dist_sq = 1.0 / (dist * dist)
    np.fill_diagonal(inv_dist_sq, 0.0)  # No self-interaction
    
    # grad[i] = -sum_j inv_dist_sq[i,j] * diff[i,j]
    # diff[i,j] = x_i - x_j
    # So grad = - (inv_dist_sq @ diff) but we need to sum over j for each i
    # Actually: grad[i] = -sum_j inv_dist_sq[i,j] * (x_i - x_j)
    # = -x_i * sum_j inv_dist_sq[i,j] + sum_j inv_dist_sq[i,j] * x_j
    # Let S[i] = sum_j inv_dist_sq[i,j]
    # Let W[i] = sum_j inv_dist_sq[i,j] * x_j
    # grad[i] = -x_i * S[i] + W[i]
    
    S = np.sum(inv_dist_sq, axis=1)  # (N,)
    W = inv_dist_sq @ x  # (N, 3)
    grad = -x * S[:, np.newaxis] + W
    
    return energy, grad

def _project_and_normalize(x):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return x / norms

def _fmin_on_sphere(x0, maxiter=10000, tol=1e-12):
    """
    Minimize energy on the sphere using a projected gradient method or L-BFGS-B.
    We'll use L-BFGS-B on the unconstrained problem with a penalty or use a sphere-constrained approach.
    
    Actually, for sphere constraints, a common approach is to use the fact that the gradient on the manifold
    is the projection of the Euclidean gradient: grad_manifold = grad_euclidean - (grad_euclidean . x_i) * x_i
    Then use a Riemannian optimization method.
    
    Alternatively, we can use L-BFGS-B on the full space and periodically project back to the sphere.
    This is a heuristic but often works well.
    
    Let's use a simple approach: use L-BFGS-B on the energy function, and after each iteration,
    project the points back to the sphere. We'll do this by defining a wrapper function.
    """
    x = x0.copy()
    x = _project_and_normalize(x)
    
    def objective_and_grad(x_flat):
        x_2d = x_flat.reshape(N, 3)
        # Project to sphere
        x_2d = _project_and_normalize(x_2d)
        e, g = _energy_and_grad(x_2d)
        # Project gradient to tangent space
        g_tangent = g - (g * x_2d).sum(axis=1, keepdims=True) * x_2d
        return e, g_tangent.flatten()
    
    res = minimize(
        objective_and_grad,
        x.flatten(),
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': tol, 'gtol': 1e-8, 'maxls': 50}
    )
    
    x_opt = _project_and_normalize(res.x.reshape(N, 3))
    return x_opt

def _energy(x):
    """Compute energy only."""
    e, _ = _energy_and_grad(x)
    return e

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_x = x0.copy()
    best_e = _energy(best_x)
    
    # Try multiple random initializations and optimize each
    np.random.seed(42)
    n_restarts = 20
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x_init = x0.copy()
        else:
            # Random points on sphere
            x_init = np.random.randn(N, 3)
            x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
        
        try:
            x_opt = _fmin_on_sphere(x_init, maxiter=5000, tol=1e-10)
            e_opt = _energy(x_opt)
            
            if e_opt < best_e:
                best_e = e_opt
                best_x = x_opt.copy()
        except Exception:
            continue
    
    # Final high-precision polish
    if time.time() < deadline - 10.0:
        try:
            x_final = _fmin_on_sphere(best_x, maxiter=20000, tol=1e-14)
            e_final = _energy(x_final)
            if e_final < best_e:
                best_x = x_final
        except Exception:
            pass
    
    # Ensure points are on unit sphere and valid
    best_x = _project_and_normalize(best_x)
    
    # Check for very close points
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_x[i] - best_x[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 1e-9:
        # This should not happen, but if it does, return the Fibonacci spiral
        return x0
    
    return best_x
# EVOLVE-BLOCK-END
