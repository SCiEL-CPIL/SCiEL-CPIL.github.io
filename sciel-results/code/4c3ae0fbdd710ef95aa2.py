"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 150

# EVOLVE-BLOCK-START

def _log_energy_and_grad(pts):
    """
    Compute the logarithmic energy E = sum_{i<j} log(1/||x_i - x_j||)
    and its gradient with respect to pts (N x 3 array on the sphere).
    
    Energy: sum_{i<j} -log(d_ij)
    Gradient wrt x_i: sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2  ... wait, let's derive.
    
    E = sum_{i<j} log(1/d_ij) = - sum_{i<j} log(d_ij)
    d_ij = ||x_i - x_j||
    
    dE/dx_i = - sum_{j != i, j>i} (1/d_ij) * d(d_ij)/dx_i  - sum_{j != i, j<i} (1/d_ij) * d(d_ij)/dx_i
    Actually, E = - sum_{i<j} log(||x_i - x_j||)
    
    dE/dx_i = - sum_{j != i} (1/2) * (1/||x_i - x_j||^2) * 2*(x_i - x_j)
             = - sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
             = sum_{j != i} (x_j - x_i) / ||x_i - x_j||^2
    
    Wait, let's be careful.
    E = sum_{i<j} -log(||x_i - x_j||)
    
    dE/dx_i = sum_{j>i} - (1/||x_i - x_j||) * (1/||x_i - x_j||) * (x_i - x_j)
            + sum_{j<i} - (1/||x_j - x_i||) * (1/||x_j - x_i||) * (-1) * (x_j - x_i) ... no.
    
    Let's use symmetry. The term involving x_i is -log(||x_i - x_j||) for all j != i.
    d/dx_i [-log(||x_i - x_j||)] = -(1/||x_i - x_j||) * (x_i - x_j)/||x_i - x_j|| = -(x_i - x_j)/||x_i - x_j||^2
    
    So grad_i E = sum_{j != i} -(x_i - x_j)/||x_i - x_j||^2 = sum_{j != i} (x_j - x_i)/||x_i - x_j||^2
    
    This is the force on x_i. To minimize E, we move in direction -grad_i E.
    But we are on the sphere, so we project the gradient onto the tangent space.
    The Riemannian gradient is (I - x_i x_i^T) * grad_i E.
    
    Let's compute the energy and the Euclidean gradient. The optimizer will handle the constraint
    or we can project. Using scipy.optimize.minimize with 'L-BFGS-B' doesn't handle sphere constraints well.
    Better to use a Riemannian optimization approach or parameterize.
    
    Actually, a simpler approach: use scipy.optimize.minimize with the constraint that ||x_i|| = 1.
    But L-BFGS-B doesn't support equality constraints well.
    
    Alternative: use gradient descent on the sphere with projection.
    Or use the fact that we can minimize over the sphere by projecting gradients.
    
    Let's implement a simple Riemannian gradient descent with backtracking line search.
    """
    # Compute pairwise distances
    # pts is N x 3
    dists = cdist(pts, pts, 'euclidean')
    # Avoid division by zero
    np.fill_diagonal(dists, 1.0)
    
    # Energy
    # E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # Using the distance matrix, sum over upper triangle
    iu = np.triu_indices(N, 1)
    E = -np.sum(np.log(dists[iu]))
    
    # Gradient
    # grad_i = sum_{j != i} (x_j - x_i) / d_ij^2
    # We can compute this using matrix operations.
    # Let's compute the matrix G where G[i,j] = (x_j - x_i) / d_ij^2
    # Actually, it's easier to compute pairwise.
    
    # dists is N x N
    # inv_d2 = 1 / dists^2
    inv_d2 = 1.0 / (dists * dists)
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad_i = sum_j (x_j - x_i) * inv_d2[i,j]
    # = (sum_j x_j * inv_d2[i,j]) - x_i * (sum_j inv_d2[i,j])
    # sum_j x_j * inv_d2[i,j] = inv_d2 @ pts  (row i)
    # sum_j inv_d2[i,j] = inv_d2.sum(axis=1)
    
    sum_x = inv_d2 @ pts
    sum_inv_d2 = inv_d2.sum(axis=1)
    
    grad = sum_x - pts * sum_inv_d2[:, np.newaxis]
    
    return E, grad


def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _riemannian_grad_descent(pts, max_iter=5000, tol=1e-10, time_limit=120.0):
    """
    Riemannian gradient descent on the sphere.
    """
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    E, grad = _log_energy_and_grad(pts)
    best_E = E
    best_pts = pts.copy()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit - 5.0:
            break
        
        # Project gradient to tangent space
        # rgrad = grad - (grad . x_i) x_i
        grad_dot = np.sum(grad * pts, axis=1, keepdims=True)
        rgrad = grad - grad_dot * pts
        
        # Check convergence
        rgrad_norm = np.linalg.norm(rgrad)
        if rgrad_norm < tol:
            break
        
        # Backtracking line search
        alpha = 0.01
        for _ in range(30):
            # Move in direction -rgrad
            new_pts = pts - alpha * rgrad
            new_pts = _project_to_sphere(new_pts)
            new_E, _ = _log_energy_and_grad(new_pts)
            
            if new_E < E:
                pts = new_pts
                E = new_E
                if E < best_E:
                    best_E = E
                    best_pts = pts.copy()
                break
            alpha *= 0.5
        
        if alpha < 1e-10:
            break
    
    return best_pts, best_E


def construct():
    start_time = time.time()
    time_limit = 120.0
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Riemannian gradient descent
    pts, E = _riemannian_grad_descent(pts, max_iter=10000, time_limit=time_limit)
    
    # If we have time, try a few random restarts with local optimization
    elapsed = time.time() - start_time
    if elapsed < time_limit - 30.0:
        best_E = E
        best_pts = pts.copy()
        
        n_restarts = 5
        for r in range(n_restarts):
            if time.time() - start_time > time_limit - 10.0:
                break
            
            # Random perturbation
            noise = np.random.randn(N, 3) * 0.1
            pts_init = _project_to_sphere(pts + noise)
            
            pts_r, E_r = _riemannian_grad_descent(pts_init, max_iter=5000, time_limit=time_limit - (time.time() - start_time) - 5.0)
            
            if E_r < best_E:
                best_E = E_r
                best_pts = pts_r.copy()
        
        return best_pts
    
    return pts

# EVOLVE-BLOCK-END
