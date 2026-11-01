"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 100

# EVOLVE-BLOCK-START

def _energy_and_grad(pts):
    """
    Compute the logarithmic energy and its gradient.
    Energy: sum_{i<j} log(1 / |x_i - x_j|)
    Gradient: for point i, grad_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    (since d/dx_i log(1/|x_i-x_j|) = (x_i-x_j)/|x_i-x_j|^2)
    """
    n = len(pts)
    # Compute pairwise distances
    # diff[i,j] = x_i - x_j
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    dist = np.sqrt(dist_sq)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist, 1.0)
    np.fill_diagonal(dist_sq, 1.0)
    
    # Energy: sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = np.sum(-np.log(dist[iu]))
    
    # Gradient: grad_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # diff[i,j] / dist_sq[i,j]
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _lbfgs_sphere(pts, max_iter=5000, tol=1e-14):
    """
    L-BFGS optimization on the sphere.
    We parameterize using the tangent space: at each step, compute gradient,
    project to tangent space, take step, then project back to sphere.
    """
    pts = _project_to_sphere(pts.copy())
    n = len(pts)
    
    # L-BFGS parameters
    m = 20  # memory
    history_p = []
    history_g = []
    
    # Compute initial energy and gradient
    energy, grad = _energy_and_grad(pts)
    
    # Project gradient to tangent space: grad_t = grad - (grad . x) * x
    grad_t = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
    
    # Line search parameters
    c1 = 1e-4
    rho = 0.5
    
    for it in range(max_iter):
        # Check convergence
        if np.max(np.abs(grad_t)) < tol:
            break
        
        # L-BFGS two-loop recursion to compute search direction
        s = history_p
        y = history_g
        if len(s) == 0:
            direction = -grad_t
        else:
            alpha = np.zeros(len(s))
            q = grad_t
            for i in range(len(s) - 1, -1, -1):
                alpha[i] = np.sum(s[i] * q) / np.sum(y[i] * y[i])
                q = q - alpha[i] * y[i]
            r = q / np.sum(q * q)
            for i in range(len(s)):
                beta = np.sum(y[i] * r) / np.sum(y[i] * y[i])
                r = r + (alpha[i] - beta) * s[i]
            direction = -r
        
        # Ensure direction is in tangent space
        direction = direction - np.sum(direction * pts, axis=1, keepdims=True) * pts
        
        # Line search: find step size alpha that satisfies Armijo condition
        step = 0.01
        energy_new, grad_new = _energy_and_grad(pts + step * direction)
        grad_new_t = grad_new - np.sum(grad_new * (pts + step * direction), axis=1, keepdims=True) * (pts + step * direction)
        
        while energy_new > energy + c1 * step * np.sum(grad_t * direction):
            step *= rho
            if step < 1e-16:
                break
            pts_new = pts + step * direction
            energy_new, grad_new = _energy_and_grad(pts_new)
            grad_new_t = grad_new - np.sum(grad_new * pts_new, axis=1, keepdims=True) * pts_new
        
        # Update
        s_vec = step * direction
        pts_new = pts + s_vec
        energy_new, grad_new = _energy_and_grad(pts_new)
        grad_new_t = grad_new - np.sum(grad_new * pts_new, axis=1, keepdims=True) * pts_new
        
        y_vec = grad_new_t - grad_t
        sy = np.sum(s_vec * y_vec)
        
        if sy > 1e-16:
            history_p.append(s_vec)
            history_g.append(y_vec)
            if len(history_p) > m:
                history_p.pop(0)
                history_g.pop(0)
        
        pts = pts_new
        energy = energy_new
        grad = grad_new
        grad_t = grad_new_t
    
    return pts, energy


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_pts = None
    best_energy = np.inf
    
    # Multiple random starts with different seeds
    rng = np.random.RandomState(42)
    
    # Start 1: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Optimize
    pts_opt, energy = _lbfgs_sphere(pts, max_iter=3000, tol=1e-14)
    if energy < best_energy:
        best_energy = energy
        best_pts = pts_opt
    
    # Start 2: Random points
    for trial in range(3):
        if time.time() > deadline:
            break
        pts = rng.randn(N, 3)
        pts = _project_to_sphere(pts)
        pts_opt, energy = _lbfgs_sphere(pts, max_iter=3000, tol=1e-14)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts_opt
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 10.0:
        pts_opt, energy = _lbfgs_sphere(best_pts, max_iter=5000, tol=1e-15)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts_opt
    
    # Ensure points are on unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    return best_pts

# EVOLVE-BLOCK-END
