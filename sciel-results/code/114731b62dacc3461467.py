"""Optimized logarithmic energy minimization on the sphere for N=150 points."""
import numpy as np
import time
import math

N = 150
TIME_LIMIT = 110.0  # safety margin before 120s kill

def _energy_and_grad(pts):
    """Compute log energy and gradient for points on unit sphere.
    
    Energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient with respect to x_i: sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2 * (-1) ... 
    Actually d/dx_i [-log(|x_i - x_j|)] = -(x_i - x_j)/|x_i - x_j|^2
    But we want to minimize, so gradient of energy w.r.t. x_i is:
    sum_{j!=i} (x_i - x_j)/|x_i - x_j|^2  (this is the force pushing i away from j)
    Wait: E = sum_{i<j} -log(d_ij). dE/dx_i = sum_{j!=i} d/dx_i[-log(d_ij)] 
    = sum_{j!=i} -(x_i - x_j)/(d_ij * d_ij) ... let me be careful.
    
    d_ij = |x_i - x_j|
    d/dx_i [log(d_ij)] = (x_i - x_j) / d_ij^2
    d/dx_i [-log(d_ij)] = -(x_i - x_j) / d_ij^2
    
    So grad_i E = sum_{j!=i} -(x_i - x_j)/d_ij^2 = sum_{j!=i} (x_j - x_i)/d_ij^2
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist = np.sqrt(np.sum(diff * diff, axis=2))  # (n, n)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum over i<j of -log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each i, grad_i = sum_{j!=i} (x_j - x_i) / dist_ij^2
    # inv_dist_sq = 1/dist^2, zero out diagonal
    inv_dist_sq = np.zeros_like(dist)
    mask = ~np.eye(n, dtype=bool)
    inv_dist_sq[mask] = 1.0 / (dist[mask] ** 2)
    
    # grad[i] = sum_j inv_dist_sq[i,j] * (pts[j] - pts[i])
    # = sum_j inv_dist_sq[i,j]*pts[j] - pts[i]*sum_j inv_dist_sq[i,j]
    sum_inv = np.sum(inv_dist_sq, axis=1)  # (n,)
    grad = inv_dist_sq @ pts - pts * sum_inv[:, np.newaxis]
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _riemannian_cg(pts, max_iter=500, tol=1e-12):
    """Riemannian conjugate gradient on the sphere manifold."""
    pts = _project_to_sphere(np.array(pts, dtype=np.float64))
    n = pts.shape[0]
    
    f, g = _energy_and_grad(pts)
    g_tan = g - np.sum(g * pts, axis=1, keepdims=True) * pts  # tangent component
    
    # Initial search direction: negative gradient
    d = -g_tan
    
    x_old = pts.copy()
    f_old = f
    g_old_tan = g_tan.copy()
    
    for k in range(max_iter):
        # Line search: minimize f along geodesic from pts in direction d
        # Geodesic: x(t) = cos(t)*pts + sin(t)*d_tan, where d_tan is unit tangent
        d_norm = np.linalg.norm(d, axis=1, keepdims=True)
        d_norm = np.maximum(d_norm, 1e-15)
        d_unit = d / d_norm
        
        # Simple backtracking line search
        t = 1.0
        for ls_iter in range(50):
            x_new = np.cos(t) * pts + np.sin(t) * d_unit
            x_new = _project_to_sphere(x_new)
            f_new, _ = _energy_and_grad(x_new)
            
            if f_new < f_old:
                break
            t *= 0.5
        else:
            break
        
        f_old = f_new
        pts = x_new
        f, g = _energy_and_grad(pts)
        g_tan = g - np.sum(g * pts, axis=1, keepdims=True) * pts
        
        # Check convergence
        if np.max(np.linalg.norm(g_tan, axis=1)) < tol:
            break
        
        # Compute beta (Polak-Ribiere)
        beta_num = np.sum(g_tan * (g_tan - g_old_tan), axis=1)
        beta_den = np.sum(g_old_tan * g_old_tan, axis=1)
        beta_den = np.maximum(beta_den, 1e-15)
        beta = np.maximum(0.0, beta_num / beta_den)
        
        # New direction
        d = -g_tan + beta * d_unit * np.linalg.norm(d, axis=1, keepdims=True)
        # Re-project d to be tangent
        d = d - np.sum(d * pts, axis=1, keepdims=True) * pts
        
        g_old_tan = g_tan.copy()
    
    return pts, f


def _fibonacci_sphere(n, offset=0.0):
    """Generate n points on sphere using Fibonacci spiral with optional offset."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    """Generate n random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Start with Fibonacci spiral
    best_pts = _fibonacci_sphere(N)
    best_energy, _ = _energy_and_grad(best_pts)
    
    # First, polish the initial guess with Riemannian CG
    best_pts, best_energy = _riemannian_cg(best_pts, max_iter=300)
    
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT - 30:
        return best_pts
    
    # Basin hopping: perturb, polish, keep if better
    n_restarts = 0
    while time.time() - start_time < TIME_LIMIT - 20:
        # Create perturbed version
        # Random rotation + small perturbation
        angle = rng.uniform(0.0, 2.0 * np.pi)
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
        
        # Rodrigues rotation formula
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        
        perturbed = best_pts @ R.T
        
        # Add small random perturbation
        perturbation_scale = 0.1 / (1.0 + 0.01 * n_restarts)
        perturbed = perturbed + perturbation_scale * rng.standard_normal((N, 3))
        perturbed = _project_to_sphere(perturbed)
        
        # Polish with Riemannian CG
        polished, energy = _riemannian_cg(perturbed, max_iter=200)
        
        if energy < best_energy:
            best_pts = polished
            best_energy = energy
        
        n_restarts += 1
        
        # Check time
        if time.time() - start_time > TIME_LIMIT - 10:
            break
    
    # Final polish with more iterations
    if time.time() - start_time < TIME_LIMIT - 5:
        best_pts, best_energy = _riemannian_cg(best_pts, max_iter=1000, tol=1e-14)
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)
    min_dist = np.min(dist)
    if min_dist < 1e-9:
        # Fallback to Fibonacci
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
