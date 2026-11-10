import numpy as np
import time
import math

N = 150

def _log_energy_and_grad(points):
    """Compute log energy and its gradient for points on unit sphere."""
    n = points.shape[0]
    # Compute pairwise distances
    # points: (n, 3)
    # diff: (n, n, 3)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero for i=j
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum_{i<j} log(1/dist_ij) = -sum_{i<j} log(dist_ij)
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu[0], iu[1]]))
    
    # Gradient: d/dx_i [ -log(|x_i - x_j|) ] = -(x_i - x_j) / |x_i - x_j|^2
    # For each pair (i,j), contribution to grad[i] is -(x_i - x_j)/dist^2
    # and to grad[j] is +(x_i - x_j)/dist^2
    grad = np.zeros_like(points)
    for i in range(n):
        for j in range(i+1, n):
            d = diff[i, j]  # x_i - x_j
            d2 = dist_sq[i, j]
            if d2 > 1e-30:
                coeff = -1.0 / d2
                grad[i] += coeff * d
                grad[j] -= coeff * d
    
    return energy, grad


def _project_to_sphere(points):
    """Project points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms


def _log_energy_fast(points):
    """Fast energy computation for tracking."""
    n = points.shape[0]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return -np.sum(np.log(dist[iu[0], iu[1]]))


def construct():
    deadline = time.time() + 110.0  # 10s safety margin
    
    # Initial: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Try multiple random restarts and keep the best
    best_points = points.copy()
    best_energy = _log_energy_fast(best_points)
    
    # Optimization using gradient descent with projection
    # Use a simple but effective approach: L-BFGS-B with projection
    
    from scipy.optimize import minimize
    
    def objective_and_grad(x_flat):
        pts = x_flat.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        e, g = _log_energy_and_grad(pts)
        # Gradient with respect to flat params (before projection)
        # For simplicity, use the projected gradient
        g_flat = g.reshape(-1)
        return e, g_flat
    
    # Run optimization
    try:
        result = minimize(
            objective_and_grad,
            points.reshape(-1),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.x is not None and np.all(np.isfinite(result.x)):
            cand = _project_to_sphere(result.x.reshape(-1, 3))
            cand_energy = _log_energy_fast(cand)
            if cand_energy < best_energy:
                best_energy = cand_energy
                best_points = cand.copy()
    except Exception:
        pass
    
    # If time permits, do a second round with tighter tolerances
    if time.time() < deadline:
        try:
            result2 = minimize(
                objective_and_grad,
                best_points.reshape(-1),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result2.x is not None and np.all(np.isfinite(result2.x)):
                cand2 = _project_to_sphere(result2.x.reshape(-1, 3))
                cand2_energy = _log_energy_fast(cand2)
                if cand2_energy < best_energy:
                    best_energy = cand2_energy
                    best_points = cand2.copy()
        except Exception:
            pass
    
    # Final projection and validation
    best_points = _project_to_sphere(best_points)
    
    # Ensure no points are too close
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                # Perturb slightly
                best_points[j] += 1e-8 * np.random.randn(3)
                best_points = _project_to_sphere(best_points)
    
    return best_points
