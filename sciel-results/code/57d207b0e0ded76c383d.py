"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 150

# EVOLVE-BLOCK-START
import math
import numpy as np
from scipy.optimize import minimize


def _log_energy_and_grad(pts):
    """Compute log(1/d) energy and gradient for points on unit sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient w.r.t. x_i: sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    # Pairwise distances
    # diff[i,j] = x_i - x_j
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist = np.linalg.norm(diff, axis=2)  # (n, n)
    np.fill_diagonal(dist, 1.0)  # avoid division by zero
    
    # Energy: sum over i<j of log(1/dist) = -sum_{i<j} log(dist)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each i, grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # diff[i,j] = x_i - x_j, so grad_i = sum_j diff[i,j] / dist[i,j]^2
    inv_dist_sq = np.zeros_like(dist)
    np.fill_diagonal(inv_dist_sq, 0.0)
    inv_dist_sq = 1.0 / (dist * dist)
    # grad[i] = sum_j diff[i,j] * inv_dist_sq[i,j]
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _halton_init(n, dims=3):
    """Generate Halton sequence points on the unit sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    
    points = np.zeros((n, dims))
    for i in range(n):
        for j in range(dims):
            base = j + 2  # 2, 3, 5, ...
            points[i, j] = halton(i + 1, base)
    
    # Map [0,1)^3 to sphere via Fibonacci-like mapping
    # Use a better mapping: convert to spherical coords
    z = 1.0 - 2.0 * points[:, 0]  # z in (-1, 1]
    theta = 2.0 * np.pi * points[:, 1]
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
    return pts


def _fibonacci_spiral(n):
    """Golden-angle Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _energy_only(pts):
    """Just compute energy for quick evaluation."""
    e, _ = _log_energy_and_grad(pts)
    return e


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    # Start with Fibonacci spiral
    pts = _fibonacci_spiral(N)
    pts = _project_to_sphere(pts)
    best_pts = pts.copy()
    best_energy = _energy_only(pts)
    
    # Try Halton init too
    halton_pts = _halton_init(N)
    halton_energy = _energy_only(halton_pts)
    if halton_energy < best_energy:
        best_pts = halton_pts.copy()
        best_energy = halton_energy
    
    # Gradient-based optimization with projected gradient
    # We'll use a simple projected gradient descent with line search
    # Or use scipy's L-BFGS-B with sphere constraint handled by projection
    
    # Strategy: Use L-BFGS-B on the sphere via parameterization or projected gradient
    # For 150 points, we have 450 variables. Let's use projected gradient descent.
    
    def projected_grad_step(x, lr):
        """One step of projected gradient descent."""
        e, g = _log_energy_and_grad(x)
        # Project gradient to tangent space
        x_norm = np.linalg.norm(x, axis=1, keepdims=True)
        x_unit = x / x_norm
        # Tangent projection: g_tangent = g - (g . x_unit) * x_unit
        g_dot_x = np.sum(g * x_unit, axis=1, keepdims=True)
        g_tangent = g - g_dot_x * x_unit
        # Update
        x_new = x - lr * g_tangent
        # Project back to sphere
        x_new = _project_to_sphere(x_new)
        return x_new, e
    
    # Adaptive learning rate projected gradient descent
    lr = 0.01
    prev_energy = best_energy
    patience = 0
    max_patience = 200
    
    while time.time() < deadline - 5.0:
        x_new, e_new = projected_grad_step(pts, lr)
        
        if e_new < prev_energy - 1e-15:
            # Accept step
            pts = x_new
            prev_energy = e_new
            patience = 0
            # Slightly increase lr if making good progress
            if patience == 0 and lr < 0.1:
                lr = min(lr * 1.05, 0.1)
        else:
            # Reject step, decrease lr
            lr *= 0.8
            patience += 1
        
        if e_new < best_energy:
            best_energy = e_new
            best_pts = pts.copy()
        
        if patience > max_patience:
            break
    
    # Final polish: use scipy L-BFGS-B with careful handling
    # Parameterize using tangent space or just do more projected gradient with smaller lr
    if time.time() < deadline - 10.0:
        # Fine-tuning with very small steps
        lr = 0.001
        prev_e = best_energy
        for _ in range(500):
            if time.time() > deadline - 5.0:
                break
            pts = _project_to_sphere(best_pts)
            e, g = _log_energy_and_grad(pts)
            x_norm = np.linalg.norm(pts, axis=1, keepdims=True)
            x_unit = pts / x_norm
            g_dot_x = np.sum(g * x_unit, axis=1, keepdims=True)
            g_tangent = g - g_dot_x * x_unit
            pts_new = _project_to_sphere(pts - lr * g_tangent)
            e_new = _energy_only(pts_new)
            if e_new < e - 1e-15:
                pts = pts_new
                if e_new < best_energy:
                    best_energy = e_new
                    best_pts = pts.copy()
                lr = min(lr * 1.02, 0.01)
            else:
                lr *= 0.9
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    
    # Validate: no points closer than 1e-9
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)
    if np.min(dist) < 1e-9:
        # Fallback to Fibonacci spiral
        best_pts = _fibonacci_spiral(N)
        best_pts = _project_to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
