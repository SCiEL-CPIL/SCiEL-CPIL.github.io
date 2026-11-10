"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math

N = 150

# EVOLVE-BLOCK-START
def _log_energy(points):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    # Compute pairwise distances
    # Use broadcasting for efficiency
    # points shape: (n, 3)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dist, 1.0)
    # Compute log(1/dist) = -log(dist)
    # Only sum upper triangle (i < j)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    log_inv_dist = -np.log(dist)
    return math.fsum(log_inv_dist[mask].tolist())


def _log_energy_grad(points):
    """
    Compute gradient of logarithmic energy.
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(points)
    grad = np.zeros_like(points)
    # Compute pairwise differences and distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (n, n, 3)
    dist = np.sqrt(np.sum(diff * diff, axis=2))  # (n, n)
    np.fill_diagonal(dist, 1.0)
    # Avoid division by zero on diagonal
    inv_dist2 = np.zeros_like(dist)
    off_diag = ~np.eye(n, dtype=bool)
    inv_dist2[off_diag] = 1.0 / dist[off_diag]
    # grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # This is: sum_j diff[i,j] * inv_dist2[i,j]
    grad = np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)
    return grad


def _project_to_sphere(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _riemannian_gradient(points, euclidean_grad):
    """
    Project Euclidean gradient onto tangent space of sphere.
    Tangent space at x: {v : x^T v = 0}
    Projection: v - (x^T v) x
    """
    return euclidean_grad - np.sum(points * euclidean_grad, axis=1, keepdims=True) * points


def _retract(points, direction, step):
    """
    Retraction: move along direction then normalize.
    p_new = normalize(p + step * direction)
    """
    new_points = points + step * direction
    return _project_to_sphere(new_points)


def _line_search(points, direction, initial_step, max_iter=50):
    """
    Geodesic line search with backtracking.
    Finds step size such that energy decreases sufficiently.
    """
    f0 = _log_energy(points)
    g = _riemannian_gradient(points, _log_energy_grad(points))
    # Directional derivative
    dg = np.sum(g * direction)
    if dg >= 0:
        # Direction is not a descent direction; project it
        # Find component in descent direction
        # For now, just try small steps
        step = initial_step
    else:
        step = initial_step
    c1 = 1e-4
    for _ in range(max_iter):
        new_points = _retract(points, direction, step)
        f1 = _log_energy(new_points)
        if f1 < f0 + c1 * step * dg:
            return new_points, step
        step *= 0.5
    return new_points, step


def construct():
    """
    Construct 150 points on unit sphere minimizing logarithmic energy.
    Uses Riemannian conjugate gradient on the sphere.
    """
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = _project_to_sphere(points)
    
    # Riemannian conjugate gradient
    max_iter = 2000
    tol = 1e-12
    
    p = _riemannian_gradient(points, _log_energy_grad(points))
    # Initial search direction
    d = -p
    # Parallel transport p from previous iteration
    p_old = p.copy()
    
    best_points = points.copy()
    best_energy = _log_energy(points)
    
    for it in range(max_iter):
        # Check convergence
        g = _log_energy_grad(points)
        rg = _riemannian_gradient(points, g)
        grad_norm = np.linalg.norm(rg)
        if grad_norm < tol:
            break
        
        # Line search
        new_points, step = _line_search(points, d, 0.1)
        
        # Compute new gradient
        g_new = _log_energy_grad(new_points)
        rg_new = _riemannian_gradient(new_points, g_new)
        
        # Parallel transport p to new point
        # For sphere: p_new = p - (p^T (new - old)) / |new - old|^2 * (new - old)
        # More accurately: parallel transport along geodesic
        # Simple approximation: p_transport = p - 2*(p^T d) * d / |d|^2 ... 
        # Actually, for retraction, use: p_new = p - (p^T (new_points - points)) / |new_points - points|^2 * (new_points - points)
        diff_vec = new_points - points
        diff_norm2 = np.sum(diff_vec * diff_vec, axis=1, keepdims=True)
        diff_norm2 = np.maximum(diff_norm2, 1e-15)
        p_transport = p_old - np.sum(p_old * diff_vec, axis=1, keepdims=True) / diff_norm2 * diff_vec
        
        # Compute beta (Polak-Ribiere)
        beta_num = np.sum(rg_new * (rg_new - p_transport))
        beta_den = np.sum(p_transport * p_transport)
        beta = max(0.0, beta_num / beta_den) if beta_den > 1e-15 else 0.0
        
        # New search direction
        d_new = -rg_new + beta * p_transport
        
        # If not descent, restart
        if np.sum(d_new * rg_new) > 0:
            d_new = -rg_new
        
        points = new_points
        p_old = p_transport
        p = rg_new
        d = d_new
        
        # Track best
        current_energy = _log_energy(points)
        if current_energy < best_energy:
            best_energy = current_energy
            best_points = points.copy()
    
    # Final normalization
    return _project_to_sphere(best_points)
# EVOLVE-BLOCK-END
