"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def _energy(points):
    """Logarithmic energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)."""
    # points: (N, 3) array on unit sphere
    # Compute pairwise distances
    # d_ij^2 = 2 - 2 * x_i . x_j
    dots = points @ points.T
    # clip for numerical stability
    dots = np.clip(dots, -1.0, 1.0)
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 0.0)
    # Only upper triangle
    iu = np.triu_indices(N, 1)
    d2_u = d2[iu]
    d2_u = np.maximum(d2_u, 1e-30)
    d_u = np.sqrt(d2_u)
    terms = -np.log(d_u)
    return math.fsum(terms.tolist())


def _grad_energy(points):
    """Gradient of logarithmic energy w.r.t. points.
    Energy = -sum_{i<j} log(|x_i - x_j|)
    dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    # points: (N, 3)
    # Compute pairwise differences and distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (N, N, 3)
    d2 = np.sum(diff * diff, axis=2)  # (N, N)
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(d2)
    # factor: 1/d^2
    factor = 1.0 / (d2)
    np.fill_diagonal(factor, 0.0)
    # grad_i = sum_j (x_i - x_j) / d_ij^2
    # diff[i,j] = x_i - x_j
    grad = np.einsum('ij,ijk->ik', factor, diff)
    return grad


def _project_to_sphere(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _optimize_from_start(start_points, time_limit):
    """Run L-BFGS-B style optimization on the sphere using retraction."""
    # We'll do gradient descent with line search on the sphere
    # Flatten the points
    x = start_points.flatten().copy()
    n = len(x)
    
    # Convert to/from 3D
    def to_3d(flat):
        return flat.reshape(N, 3)
    
    def from_3d(pts):
        return pts.flatten()
    
    def fun(flat):
        pts = to_3d(flat)
        pts = _project_to_sphere(pts)
        return _energy(pts)
    
    def grad(flat):
        pts = to_3d(flat)
        pts = _project_to_sphere(pts)
        g = _grad_energy(pts)
        # Project gradient onto tangent space: g - (g.x)x
        g_tan = g - (g * pts).sum(axis=1, keepdims=True) * pts
        return g_tan.flatten()
    
    best_x = x.copy()
    best_f = fun(x)
    t0 = time.time()
    
    # L-BFGS with retraction: use scipy's minimize with custom constraints
    # Actually, let's use a simple gradient descent with Armijo line search
    # on the sphere via retraction
    
    lr = 0.01
    for step in range(5000):
        if time.time() - t0 > time_limit:
            break
        pts = to_3d(best_x)
        pts = _project_to_sphere(pts)
        g = _grad_energy(pts)
        g_tan = g - (g * pts).sum(axis=1, keepdims=True) * pts
        g_norm = np.linalg.norm(g_tan)
        if g_norm < 1e-10:
            break
        
        # Armijo line search
        alpha = lr
        for _ in range(20):
            # Retraction: move in tangent direction, then project
            new_pts = pts + alpha * g_tan
            new_pts = _project_to_sphere(new_pts)
            new_f = _energy(new_pts)
            if new_f < best_f - 1e-4 * alpha * g_norm * g_norm:
                break
            alpha *= 0.5
        
        if new_f < best_f:
            best_f = new_f
            best_x = from_3d(new_pts)
        else:
            # Try smaller step
            alpha = max(alpha * 0.1, 1e-8)
            new_pts = pts + alpha * g_tan
            new_pts = _project_to_sphere(new_pts)
            new_f = _energy(new_pts)
            if new_f < best_f:
                best_f = new_f
                best_x = from_3d(new_pts)
    
    return to_3d(best_x), best_f


def construct():
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    start = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = start
    best_energy = _energy(start)
    
    # Optimize from the start
    opt_points, opt_energy = _optimize_from_start(start, time_limit * 0.7)
    if opt_energy < best_energy:
        best_energy = opt_energy
        best_points = opt_points
    
    # If we have time, try a second restart with slightly perturbed start
    elapsed = time.time() - start_time
    if elapsed < time_limit - 5.0:
        np.random.seed(42)
        perturbed = start + np.random.randn(N, 3) * 0.05
        perturbed = _project_to_sphere(perturbed)
        opt2_points, opt2_energy = _optimize_from_start(perturbed, time_limit - elapsed - 5.0)
        if opt2_energy < best_energy:
            best_energy = opt2_energy
            best_points = opt2_points
    
    return _project_to_sphere(best_points)
# EVOLVE-BLOCK-END
