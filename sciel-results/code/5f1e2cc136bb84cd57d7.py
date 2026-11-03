"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def _energy_and_grad(pts):
    """Compute log energy and gradient for points on unit sphere.
    pts: (N, 3) array of unit vectors
    Returns: (energy, grad) where grad is (N, 3)
    """
    n = pts.shape[0]
    # Pairwise distances
    # Use broadcasting for efficiency
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist2 = np.sum(diff * diff, axis=2)  # (N, N)
    # Avoid diagonal (self-distance = 0)
    np.fill_diagonal(dist2, 1.0)
    dist = np.sqrt(dist2)
    
    # Energy: sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: d/dx_i [ -log(dist_ij) ] for i != j
    # = -(1/dist_ij) * d(dist_ij)/dx_i
    # d(dist_ij)/dx_i = (x_i - x_j) / dist_ij
    # So gradient contribution to x_i from pair (i,j) = -(x_i - x_j) / dist_ij^2
    # Total gradient for x_i = sum_{j != i} -(x_i - x_j) / dist_ij^2
    
    # Compute gradient efficiently
    # grad[i] = sum_j (x_j - x_i) / dist_ij^2  for j != i
    # = sum_j x_j / dist_ij^2 - x_i * sum_j 1/dist_ij^2
    
    inv_dist2 = 1.0 / dist2  # (N, N)
    np.fill_diagonal(inv_dist2, 0.0)
    
    # sum_j x_j / dist_ij^2
    grad = inv_dist2 @ pts  # (N, 3)
    # subtract x_i * sum_j 1/dist_ij^2
    row_sums = np.sum(inv_dist2, axis=1)  # (N,)
    grad -= pts * row_sums[:, np.newaxis]
    
    return energy, grad


def _energy_only(pts):
    """Compute just the energy for speed."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    dist = np.sqrt(dist2)
    iu = np.triu_indices(n, k=1)
    return -np.sum(np.log(dist[iu]))


def _fibonacci_spiral(n, offset=0.0, twist=1.0):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i * twist
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(n, seed=42):
    """Generate random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return pts / norms


def _adam_optimize(pts0, max_iter=5000, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, 
                   deadline=None, check_interval=50):
    """Adam optimization on the sphere.
    We optimize in Cartesian coordinates and project back to sphere after each step.
    """
    pts = pts0.copy()
    best_pts = pts.copy()
    best_energy = _energy_only(pts)
    
    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    t = 0
    
    start_time = time.time()
    
    for t in range(1, max_iter + 1):
        if deadline is not None and (t % check_interval == 0 or t == 1):
            if time.time() > deadline:
                break
        
        energy, grad = _energy_and_grad(pts)
        
        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        update = m_hat / (np.sqrt(v_hat) + eps)
        
        pts_new = pts - lr * update
        pts_new = _project_to_sphere(pts_new)
        
        # Check for improvement
        if not np.all(np.isfinite(pts_new)):
            break
        
        energy_new = _energy_only(pts_new)
        
        if energy_new < best_energy:
            best_energy = energy_new
            best_pts = pts_new.copy()
        
        pts = pts_new
    
    return best_pts, best_energy


def _nelder_mead_polish(pts, max_iter=2000, tol=1e-12):
    """Nelder-Mead polish in spherical coordinates for high precision."""
    # Convert to spherical coordinates
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    r = np.sqrt(x*x + y*y)
    theta = np.arctan2(y, x)  # azimuth
    phi = np.arccos(np.clip(z, -1.0, 1.0))  # polar
    
    params = np.column_stack([theta, phi])  # (N, 2)
    
    def energy_from_params(p):
        th = p[:, 0]
        ph = p[:, 1]
        x = np.sin(ph) * np.cos(th)
        y = np.sin(ph) * np.sin(th)
        z = np.cos(ph)
        pts = np.column_stack([x, y, z])
        return _energy_only(pts)
    
    # Use scipy's Nelder-Mead if available, otherwise simple implementation
    try:
        from scipy.optimize import minimize
        res = minimize(energy_from_params, params, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol})
        if res.success or res.fun < energy_from_params(params):
            p_opt = res.x
            th = p_opt[:, 0]
            ph = p_opt[:, 1]
            x = np.sin(ph) * np.cos(th)
            y = np.sin(ph) * np.sin(th)
            z = np.cos(ph)
            pts_opt = np.column_stack([x, y, z])
            return pts_opt
    except:
        pass
    
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 10s safety margin
    
    best_pts = None
    best_energy = float('inf')
    
    # Generate multiple starting configurations
    starts = []
    
    # Fibonacci spiral (original)
    starts.append(_fibonacci_spiral(N))
    
    # Fibonacci spiral with different offsets
    for off in [0.1, 0.2, 0.3, 0.5, 0.7]:
        starts.append(_fibonacci_spiral(N, offset=off))
    
    # Fibonacci spiral with twist variations
    for tw in [0.8, 1.2, 1.5, 2.0]:
        starts.append(_fibonacci_spiral(N, twist=tw))
    
    # Random starts
    for seed in [1, 2, 3, 7, 13, 42, 100, 200]:
        starts.append(_random_points(N, seed=seed))
    
    # Try each start with Adam optimization
    for idx, pts0 in enumerate(starts):
        if time.time() > deadline - 20.0:  # Need time for at least one more full optimization
            break
        
        # Check if we have time for a full optimization
        elapsed = time.time() - start_time
        if elapsed > 110.0:
            break
        
        pts_opt, energy_opt = _adam_optimize(pts0, max_iter=3000, lr=0.005, 
                                              deadline=deadline - 30.0)
        
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt.copy()
    
    # Polish the best result
    if best_pts is not None and time.time() < deadline - 10.0:
        best_pts = _nelder_mead_polish(best_pts, max_iter=1000, tol=1e-14)
        best_energy = _energy_only(best_pts)
    
    # Final check
    if best_pts is None:
        # Fallback to Fibonacci spiral
        best_pts = _fibonacci_spiral(N)
    
    # Ensure finite values and proper shape
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    best_pts = best_pts / norms
    
    return best_pts
# EVOLVE-BLOCK-END
