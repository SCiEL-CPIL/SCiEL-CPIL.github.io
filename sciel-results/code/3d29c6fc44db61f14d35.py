"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

DEADLINE_SECONDS = 115.0  # Safety margin before 150s kill (task says 120s, but let's be safe)

def _energy_and_grad(pts):
    """Compute logarithmic energy E = sum_{i<j} log(1/|xi-xj|) and its gradient."""
    n = pts.shape[0]
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist = np.linalg.norm(diff, axis=2)  # (n, n)
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum over i<j of log(1/dist[i,j]) = -sum over i<j of log(dist[i,j])
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dist[iu])
    energy = -np.sum(log_dists)
    
    # Gradient: dE/dx_i = - sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    dist2 = dist * dist
    np.fill_diagonal(dist2, 1.0)
    
    inv_dist2 = 1.0 / dist2
    inv_dist2[np.diag_indices(n)] = 0.0
    
    # diff[i,j] = pts[i] - pts[j]
    # grad[i] = - sum_j diff[i,j] / dist2[i,j]
    grad = -np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms


def _fibonacci_spiral(n):
    """Generate points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts


def _random_points(n, rng):
    """Generate random points on sphere."""
    pts = rng.standard_normal((n, 3))
    return _project_to_sphere(pts)


def _optimize_with_lbfgs(pts, time_limit=100.0):
    """Optimize using L-BFGS-B with spherical constraint handled by projection."""
    start_time = time.time()
    
    def fun_and_grad(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        energy, grad = _energy_and_grad(pts)
        # Compute gradient in ambient space, then project to tangent space
        # grad_tangent = grad - (grad . x) x
        grad_ambient = grad.reshape(-1)
        # Project gradient to tangent space
        pts_flat = pts.reshape(-1)
        # For each point, project its gradient
        grad_tangent = np.zeros_like(grad_ambient)
        for i in range(N):
            p = pts[i]
            g = grad[i]
            # Tangent projection: g - (g.p)p
            grad_tangent[3*i:3*i+3] = g - np.dot(g, p) * p
        return energy, grad_tangent
    
    x0 = pts.reshape(-1)
    
    # Use L-BFGS-B
    result = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10,
                               'maxfun': 500, 'disp': False})
    
    elapsed = time.time() - start_time
    best_pts = _project_to_sphere(result.x.reshape(N, 3))
    best_energy, _ = _energy_and_grad(best_pts)
    
    return best_pts, best_energy, elapsed


def _optimize_with_gradient_descent(pts, time_limit=100.0):
    """Optimize using gradient descent with adaptive step size."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    best_energy, _ = _energy_and_grad(pts)
    best_pts = pts.copy()
    
    step = 0.01
    min_step = 1e-8
    max_iter = 10000
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        energy, grad = _energy_and_grad(pts)
        
        # Project gradient to tangent space
        grad_tangent = np.zeros_like(pts)
        for i in range(N):
            grad_tangent[i] = grad[i] - np.dot(grad[i], pts[i]) * pts[i]
        
        # Update best
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        # Try a step
        new_pts = _project_to_sphere(pts - step * grad_tangent)
        new_energy, _ = _energy_and_grad(new_pts)
        
        if new_energy < energy:
            pts = new_pts
            # Increase step slightly if improvement is good
            if new_energy < energy - 1e-8:
                step = min(step * 1.05, 0.1)
        else:
            # Decrease step
            step *= 0.9
            if step < min_step:
                break
        
        # Convergence check
        if np.linalg.norm(grad_tangent) < 1e-12:
            break
    
    return best_pts, best_energy, time.time() - start_time


def _polish_with_lbfgs(pts, time_limit=30.0):
    """Final polish with high-precision L-BFGS."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    def fun_and_grad(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        energy, grad = _energy_and_grad(pts)
        grad_tangent = np.zeros_like(grad)
        for i in range(N):
            p = pts[i]
            g = grad[i]
            grad_tangent[i] = g - np.dot(g, p) * p
        return energy, grad_tangent.reshape(-1)
    
    x0 = pts.reshape(-1)
    
    result = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                      options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12,
                               'maxfun': 200, 'disp': False})
    
    best_pts = _project_to_sphere(result.x.reshape(N, 3))
    best_energy, _ = _energy_and_grad(best_pts)
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Multiple initializations
    rng = np.random.default_rng(42)
    
    # Initialization 1: Fibonacci spiral
    pts1 = _fibonacci_spiral(N)
    
    # Initialization 2: Random points
    pts2 = _random_points(N, rng)
    
    # Initialization 3: Slightly perturbed Fibonacci
    pts3 = _project_to_sphere(_fibonacci_spiral(N) + 0.01 * rng.standard_normal((N, 3)))
    
    # Optimization strategies
    candidates = []
    
    # Strategy 1: L-BFGS from Fibonacci
    elapsed = DEADLINE_SECONDS - (time.time() - start_time)
    if elapsed > 5:
        pts_opt, energy_opt, elapsed_used = _optimize_with_lbfgs(pts1, time_limit=elapsed - 5)
        candidates.append((pts_opt, energy_opt))
    
    # Strategy 2: Gradient descent from Fibonacci
    elapsed = DEADLINE_SECONDS - (time.time() - start_time)
    if elapsed > 5:
        pts_opt, energy_opt, elapsed_used = _optimize_with_gradient_descent(pts1, time_limit=elapsed - 5)
        candidates.append((pts_opt, energy_opt))
    
    # Strategy 3: L-BFGS from random
    elapsed = DEADLINE_SECONDS - (time.time() - start_time)
    if elapsed > 5:
        pts_opt, energy_opt, elapsed_used = _optimize_with_lbfgs(pts2, time_limit=elapsed - 5)
        candidates.append((pts_opt, energy_opt))
    
    # Strategy 4: Gradient descent from random
    elapsed = DEADLINE_SECONDS - (time.time() - start_time)
    if elapsed > 5:
        pts_opt, energy_opt, elapsed_used = _optimize_with_gradient_descent(pts2, time_limit=elapsed - 5)
        candidates.append((pts_opt, energy_opt))
    
    # Strategy 5: L-BFGS from perturbed Fibonacci
    elapsed = DEADLINE_SECONDS - (time.time() - start_time)
    if elapsed > 5:
        pts_opt, energy_opt, elapsed_used = _optimize_with_lbfgs(pts3, time_limit=elapsed - 5)
        candidates.append((pts_opt, energy_opt))
    
    # Select best candidate
    if not candidates:
        # Fallback: return Fibonacci spiral
        return _fibonacci_spiral(N)
    
    best_pts, best_energy = min(candidates, key=lambda x: x[1])
    
    # Final polish with L-BFGS
    elapsed = DEADLINE_SECONDS - (time.time() - start_time)
    if elapsed > 5:
        polished_pts, polished_energy = _polish_with_lbfgs(best_pts, time_limit=elapsed - 5)
        if polished_energy < best_energy:
            best_pts = polished_pts
            best_energy = polished_energy
    
    return best_pts
# EVOLVE-BLOCK-END
