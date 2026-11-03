"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _log_energy(points):
    """Compute log(1/d) energy for points on unit sphere."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dists[iu])
    return -np.sum(log_dists)


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_uniform_sphere(n, seed=42):
    """Generate random uniform points on sphere."""
    rng = np.random.RandomState(seed)
    points = rng.randn(n, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _energy_and_grad(x):
    """Energy and gradient for L-BFGS-B. x is flat array of shape (N*3,)."""
    points = x.reshape(N, 3)
    # Project to sphere for energy computation
    points = _project_to_sphere(points)
    
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    
    # Energy = -sum_{i<j} log(d_ij) = -0.5 * sum_{i!=j} log(d_ij)
    # For gradient, we work with the full sum then halve
    # E = -sum_{i<j} log(d_ij)
    # dE/dx_i = sum_{j!=i} (x_i - x_j) / d_ij^2  (this is the gradient of E)
    # Wait, let me re-derive:
    # E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # dE/dx_i = -sum_{j!=i, j>i or j<i} d/dx_i [log(d_ij)] * (1/2 for i<j case)
    # Actually for i<j: d/dx_i [-log(d_ij)] = -1/d_ij * (x_i-x_j)/d_ij = -(x_i-x_j)/d_ij^2
    # For j<i: d/dx_i [-log(d_ji)] = -1/d_ji * (x_i-x_j)/d_ji = -(x_i-x_j)/d_ij^2
    # So total: dE/dx_i = -sum_{j!=i} (x_i-x_j)/d_ij^2
    # But wait, that's the gradient of E, and we want to minimize E, so L-BFGS will use this.
    
    # Let me be more careful. E = -sum_{i<j} log(d_ij)
    # For a fixed pair (i,j) with i<j:
    # dE/dx_i = -d/dx_i log(d_ij) = -(1/d_ij) * d(d_ij)/dx_i = -(1/d_ij) * (x_i-x_j)/d_ij = -(x_i-x_j)/d_ij^2
    # dE/dx_j = -(1/d_ij) * (x_j-x_i)/d_ij = (x_i-x_j)/d_ij^2
    # So for point i: dE/dx_i = sum_{j>i} [-(x_i-x_j)/d_ij^2] + sum_{j<i} [(x_i-x_j)/d_ij^2]
    # Wait no. Let me redo.
    # E = -sum_{i<j} log(d_ij)
    # The term involving x_i is: for each j>i, term is -log(d_ij), for each j<i, term is -log(d_ji) = -log(d_ij)
    # d/dx_i [-log(d_ij)] where d_ij = |x_i - x_j|:
    # = -(1/d_ij) * (x_i - x_j) / d_ij = -(x_i - x_j) / d_ij^2
    # So dE/dx_i = sum_{j!=i} [-(x_i - x_j) / d_ij^2]
    # = sum_{j!=i} (x_j - x_i) / d_ij^2
    
    # Compute gradient
    weights = np.zeros_like(dists_sq)
    mask = ~np.eye(n, dtype=bool)
    weights[mask] = 1.0 / dists_sq[mask]
    
    sum_weights = np.sum(weights, axis=1)
    weighted_sum = weights @ points
    
    # grad_i = sum_j weights[i,j] * (x_j - x_i) = weighted_sum[i] - x_i * sum_weights[i]
    grads = weighted_sum - points * sum_weights[:, np.newaxis]
    
    # Project gradient to tangent space
    dot_products = np.sum(grads * points, axis=1, keepdims=True)
    grads_tangent = grads - dot_products * points
    
    energy = _log_energy(points)
    return energy, grads_tangent.flatten()


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Multiple initializations
    inits = [
        _fibonacci_sphere(N),
        _random_uniform_sphere(N, seed=1),
        _random_uniform_sphere(N, seed=2),
        _random_uniform_sphere(N, seed=3),
        _random_uniform_sphere(N, seed=4),
    ]
    
    best_points = inits[0].copy()
    best_energy = _log_energy(best_points)
    
    for idx, init in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
        
        x0 = init.flatten()
        
        try:
            res = minimize(
                _energy_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 2000, 'maxfun': 20000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if res.fun < best_energy:
                p_res = _project_to_sphere(res.x.reshape(N, 3))
                e_check = _log_energy(p_res)
                if e_check < best_energy:
                    best_energy = e_check
                    best_points = p_res.copy()
        except Exception:
            pass
    
    # Final polish on best point
    if time.time() < deadline - 3.0:
        x_best = best_points.flatten()
        try:
            res = minimize(
                _energy_and_grad,
                x_best,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 3000, 'maxfun': 30000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            p_res = _project_to_sphere(res.x.reshape(N, 3))
            e_check = _log_energy(p_res)
            if e_check < best_energy:
                best_energy = e_check
                best_points = p_res.copy()
        except Exception:
            pass
    
    best_points = _project_to_sphere(best_points)
    
    if not np.all(np.isfinite(best_points)):
        return _fibonacci_sphere(N)
    
    # Check for points too close
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(N, k=1)
    if np.min(dists[iu]) < 1e-9:
        return _fibonacci_sphere(N)
    
    return best_points
# EVOLVE-BLOCK-END
