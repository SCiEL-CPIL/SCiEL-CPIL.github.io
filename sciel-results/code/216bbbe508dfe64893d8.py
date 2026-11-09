"""
Strategy: Multi-restart L-BFGS-B optimization on spherical coordinates.
- Uses spherical coordinates (theta, phi) to remove the sphere constraint.
- Implements a custom gradient for the logarithmic energy.
- Uses a deadline guard to return the best result found before 145s.
- Includes a final high-precision polish pass.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 150
TIME_LIMIT = 145.0  # Safety margin for 150s limit

def _spherical_to_cartesian(params):
    """Convert spherical coordinates (theta, phi) to Cartesian coordinates."""
    theta = params[0:N]
    phi = params[N:2*N]
    sin_theta = np.sin(theta)
    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])

def _log_energy_and_grad(params):
    """
    Compute logarithmic energy and gradient for points on the unit sphere.
    params: (2N,) array of [theta_1, ..., theta_N, phi_1, ..., phi_N]
    """
    points = _spherical_to_cartesian(params)
    
    # Compute pairwise distances
    # dist[i, j] = ||x_i - x_j||
    # dist^2 = 2 - 2 * dot(x_i, x_j)
    dots = np.dot(points, points.T)
    # Clip to avoid numerical issues
    dots = np.clip(dots, -1.0, 1.0)
    
    # Distance squared
    dist_sq = 2.0 - 2.0 * dots
    np.fill_diagonal(dist_sq, 1.0)  # Avoid division by zero on diagonal
    
    # Distances
    dist = np.sqrt(dist_sq)
    
    # Energy: sum_{i<j} log(1/d) = - sum_{i<j} log(d)
    # We only sum i<j
    iu = np.triu_indices(N, k=1)
    d_pairs = dist[iu]
    
    # Handle very small distances (should not happen with good init, but safe)
    d_pairs = np.maximum(d_pairs, 1e-15)
    
    energy = -np.sum(np.log(d_pairs))
    
    # Gradient computation
    # E = - sum_{i<j} log(d_ij)
    # dE/dx_i = - sum_{j != i} (1/d_ij) * (d_ij / dx_i)
    # d_ij / dx_i = (x_i - x_j) / d_ij
    # So dE/dx_i = - sum_{j != i} (x_i - x_j) / d_ij^2
    
    # Compute gradient in Cartesian space
    grad_cart = np.zeros_like(points)
    
    # Vectorized gradient calculation
    # We need sum_{j != i} (x_i - x_j) / d_ij^2
    # Let A[i,j] = (x_i - x_j) / d_ij^2
    # Note: d_ij^2 = dist_sq[i,j]
    
    # Create matrix of differences
    # diff[i,j] = x_i - x_j
    # This is expensive for N=150 (150x150x3), but manageable.
    
    # More efficient:
    # grad_cart_i = sum_{j != i} (x_i - x_j) / d_ij^2
    # = sum_{j != i} x_i / d_ij^2 - sum_{j != i} x_j / d_ij^2
    
    # Let's compute the matrix M where M[i,j] = 1/d_ij^2 for i != j, 0 for i=j
    inv_dist_sq = np.zeros_like(dist_sq)
    mask = np.ones((N, N), dtype=bool)
    np.fill_diagonal(mask, False)
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    
    # grad_cart[i] = sum_j inv_dist_sq[i,j] * (x_i - x_j)
    # = x_i * sum_j inv_dist_sq[i,j] - sum_j inv_dist_sq[i,j] * x_j
    
    sum_inv = np.sum(inv_dist_sq, axis=1)
    term1 = points * sum_inv[:, np.newaxis]
    term2 = np.dot(inv_dist_sq, points)
    
    grad_cart = term1 - term2
    
    # Now map gradient from Cartesian to Spherical coordinates
    # x = sin(theta) cos(phi)
    # y = sin(theta) sin(phi)
    # z = cos(theta)
    
    theta = params[0:N]
    phi = params[N:2*N]
    
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    sin_p = np.sin(phi)
    cos_p = np.cos(phi)
    
    # Jacobian of Cartesian w.r.t. Spherical
    # dx/dtheta = cos(theta)cos(phi)
    # dx/dphi = -sin(theta)sin(phi)
    # dy/dtheta = cos(theta)sin(phi)
    # dy/dphi = sin(theta)cos(phi)
    # dz/dtheta = -sin(theta)
    # dz/dphi = 0
    
    # grad_theta = grad_x * dx/dtheta + grad_y * dy/dtheta + grad_z * dz/dtheta
    # grad_phi = grad_x * dx/dphi + grad_y * dy/dphi
    
    gx = grad_cart[:, 0]
    gy = grad_cart[:, 1]
    gz = grad_cart[:, 2]
    
    grad_theta = gx * (cos_t * cos_p) + gy * (cos_t * sin_p) + gz * (-sin_t)
    grad_phi = gx * (-sin_t * sin_p) + gy * (sin_t * cos_p)
    
    # Handle singularity at poles (sin(theta) = 0)
    # If sin(theta) is very small, grad_phi is ill-defined but should be 0 due to symmetry
    # We can mask these out or just let it be, L-BFGS should handle it if init is good
    eps = 1e-12
    small_sin = np.abs(sin_t) < eps
    grad_phi[small_sin] = 0.0
    
    grad = np.concatenate([grad_theta, grad_phi])
    
    return energy, grad

def construct():
    start_time = time.time()
    best_energy = np.inf
    best_params = None
    
    # Initializations
    inits = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    points = np.column_stack([x, y, z])
    
    # Convert to spherical
    theta = np.arccos(np.clip(points[:, 2], -1.0, 1.0))
    phi_s = np.arctan2(points[:, 1], points[:, 0])
    init1 = np.concatenate([theta, phi_s])
    inits.append(init1)
    
    # 2. Random init
    np.random.seed(42)
    v = np.random.randn(N, 3)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    theta = np.arccos(np.clip(v[:, 2], -1.0, 1.0))
    phi_s = np.arctan2(v[:, 1], v[:, 0])
    init2 = np.concatenate([theta, phi_s])
    inits.append(init2)
    
    # 3. Another random init with different seed
    np.random.seed(123)
    v = np.random.randn(N, 3)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    theta = np.arccos(np.clip(v[:, 2], -1.0, 1.0))
    phi_s = np.arctan2(v[:, 1], v[:, 0])
    init3 = np.concatenate([theta, phi_s])
    inits.append(init3)
    
    # 4. Grid-like init (lat/lon)
    # Approximate grid
    lat_lines = 12
    lon_lines = 13
    # This might not give exactly 150, so we just use random perturbations of a grid
    # Let's stick to 3 inits for now to save time
    
    for init in inits:
        if time.time() - start_time > TIME_LIMIT:
            break
        
        # Run L-BFGS-B
        try:
            res = minimize(
                _log_energy_and_grad,
                init,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 2000,
                    'ftol': 1e-12,
                    'gtol': 1e-10,
                    'maxls': 50
                }
            )
            
            if res.success or res.fun < best_energy:
                # Check validity
                points = _spherical_to_cartesian(res.x)
                if np.all(np.isfinite(points)) and np.all(np.linalg.norm(points, axis=1) > 1e-9):
                    best_energy = res.fun
                    best_params = res.x
        except Exception:
            pass
    
    # Final polish: Use the best params found and run a tighter optimization
    if best_params is not None:
        if time.time() - start_time < TIME_LIMIT - 10:
            try:
                res_polish = minimize(
                    _log_energy_and_grad,
                    best_params,
                    method='L-BFGS-B',
                    jac=True,
                    options={
                        'maxiter': 5000,
                        'ftol': 1e-15,
                        'gtol': 1e-12,
                        'maxls': 100
                    }
                )
                if res_polish.fun < best_energy:
                    points = _spherical_to_cartesian(res_polish.x)
                    if np.all(np.isfinite(points)):
                        best_energy = res_polish.fun
                        best_params = res_polish.x
            except Exception:
                pass
    
    # Convert best params to Cartesian
    if best_params is None:
        # Fallback to Fibonacci if something went wrong
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    points = _spherical_to_cartesian(best_params)
    
    # Final normalization to ensure unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    return points

# EVOLVE-BLOCK-START
# (No changes needed here as the code above is the full implementation)
# EVOLVE-BLOCK-END
