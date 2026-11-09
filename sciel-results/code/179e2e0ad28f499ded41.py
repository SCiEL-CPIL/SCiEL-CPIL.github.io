import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _energy_and_grad(x):
    """
    Compute the logarithmic energy and its gradient with respect to the 3D coordinates.
    The points are normalized internally to the unit sphere.
    """
    x = x.reshape(N, 3)
    # Normalize points to unit sphere
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms < 1e-12, 1e-12, norms)
    u = x / norms
    
    # Compute pairwise distances
    # u is (N, 3)
    # diff is (N, N, 3)
    diff = u[:, np.newaxis, :] - u[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero and log(0)
    # We only need i < j
    # dist_sq is symmetric, diagonal is 0
    # We will compute energy and gradient for i < j
    
    # For gradient, we need the sum over j != i of (u_i - u_j) / ||u_i - u_j||^2
    # Let's compute the matrix of forces
    # force[i, j] = (u_i - u_j) / dist_sq[i, j]
    # But we need to be careful with i=j
    
    # dist_sq has 0 on diagonal.
    # We can compute inv_dist_sq = 1 / dist_sq, with 0 on diagonal
    inv_dist_sq = np.zeros_like(dist_sq)
    mask = dist_sq > 1e-12
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    
    # Energy: sum_{i<j} log(1/sqrt(dist_sq)) = -0.5 * sum_{i<j} log(dist_sq)
    # sum_{i<j} log(dist_sq) = 0.5 * (sum_{i!=j} log(dist_sq) - sum_{i} log(0)) 
    # Actually, sum_{i<j} log(dist_sq[i,j]) = 0.5 * sum_{i!=j} log(dist_sq[i,j])
    # So E = -0.5 * 0.5 * sum_{i!=j} log(dist_sq[i,j]) = -0.25 * sum_{i!=j} log(dist_sq[i,j])
    # Wait, E = sum_{i<j} log(1/d) = sum_{i<j} -log(d) = -sum_{i<j} log(d)
    # d = sqrt(dist_sq)
    # log(d) = 0.5 log(dist_sq)
    # E = -sum_{i<j} 0.5 log(dist_sq[i,j]) = -0.5 * sum_{i<j} log(dist_sq[i,j])
    # sum_{i<j} log(dist_sq[i,j]) = 0.5 * sum_{i!=j} log(dist_sq[i,j])
    # So E = -0.25 * sum_{i!=j} log(dist_sq[i,j])
    
    # We can compute sum_{i!=j} log(dist_sq[i,j])
    # log_dist_sq = np.log(dist_sq, where=mask, out=np.zeros_like(dist_sq))
    # sum_log = np.sum(log_dist_sq)
    # E = -0.25 * sum_log
    
    # But we need to be careful with numerical stability.
    # Let's compute E directly from the upper triangle
    iu = np.triu_indices(N, 1)
    d_sq = dist_sq[iu]
    E = np.sum(np.log(1.0 / np.sqrt(d_sq)))
    
    # Gradient
    # grad_u[i] = sum_{j != i} (u_i - u_j) / dist_sq[i, j]
    # Note: (u_i - u_j) is a vector.
    # grad_u[i] = sum_j (u_i - u_j) * inv_dist_sq[i, j]
    # This can be written as:
    # grad_u[i] = u_i * sum_j inv_dist_sq[i, j] - sum_j u_j * inv_dist_sq[i, j]
    
    # Let's compute this efficiently
    # S1[i] = sum_j inv_dist_sq[i, j]
    S1 = np.sum(inv_dist_sq, axis=1)
    
    # S2[i] = sum_j u_j * inv_dist_sq[i, j]
    # S2 = inv_dist_sq @ u
    S2 = inv_dist_sq @ u
    
    grad_u = u * S1[:, np.newaxis] - S2
    
    # Now map grad_u to grad_x
    # grad_x[i] = grad_u[i] * (1/r_i) - (grad_u[i] . u_i) * (u_i / r_i)
    # Note: u_i = x_i / r_i
    # grad_u[i] . u_i = grad_u[i] . (x_i / r_i)
    # Let's compute dot = np.sum(grad_u * u, axis=1)
    dot = np.sum(grad_u * u, axis=1)
    
    # grad_x[i] = (grad_u[i] - dot[i] * u_i) / r_i
    # Note: grad_u[i] is tangent to the sphere, so grad_u[i] . u_i = 0
    # So dot[i] should be 0 (up to numerical errors)
    # If dot[i] is 0, then grad_x[i] = grad_u[i] / r_i
    
    # Let's check if grad_u is tangent
    # It should be, by construction.
    # So grad_x = grad_u / r_i
    
    # But wait, the chain rule is:
    # dE/dx_i = dE/du_i * du_i/dx_i
    # du_i/dx_i = (1/r_i) (I - u_i u_i^T)
    # So dE/dx_i = (1/r_i) (I - u_i u_i^T) grad_u_i
    # Since grad_u_i is tangent, (I - u_i u_i^T) grad_u_i = grad_u_i
    # So dE/dx_i = grad_u_i / r_i
    
    grad_x = grad_u / norms
    
    return E, grad_x.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for 150s limit
    
    # Initial points: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z]).flatten()
    
    # Optimize
    best_x = x0
    best_E, _ = _energy_and_grad(best_x)
    
    # Use L-BFGS-B
    # We will run minimize in a loop if needed, but usually one call is enough
    # However, to be safe with the deadline, we can use a callback to check time
    
    def callback(xk):
        nonlocal best_x, best_E
        E_k, _ = _energy_and_grad(xk)
        if E_k < best_E:
            best_E = E_k
            best_x = xk.copy()
        return time.time() > deadline
    
    try:
        res = minimize(_energy_and_grad, x0, method='L-BFGS-B', jac=True, callback=callback, options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12})
        if res.success and res.fun < best_E:
            best_E = res.fun
            best_x = res.x
    except Exception as e:
        pass
    
    # Ensure points are on the sphere
    x_final = best_x.reshape(N, 3)
    norms = np.linalg.norm(x_final, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1e-12, norms)
    x_final = x_final / norms
    
    return x_final
