"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    # pts is (N, 3)
    # Compute all pairwise distances
    # Use broadcasting for efficiency
    # dist[i,j] = |pts[i] - pts[j]|
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Avoid log(0) by masking diagonal
    mask = ~np.eye(N, dtype=bool)
    # We only want i < j
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    # Clip to avoid log of very small numbers (shouldn't happen if points are distinct)
    d = np.maximum(d, 1e-15)
    return math.fsum(np.log(1.0 / d).tolist())

def _grad_log_energy(pts):
    """Compute gradient of logarithmic energy.
    E = sum_{i<j} log(1/|x_i - x_j|) = - sum_{i<j} log(|x_i - x_j|)
    dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    grad = np.zeros_like(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    inv_dist_sq = 1.0 / dist_sq
    # grad[i] = sum_j (x_i - x_j) / |x_i - x_j|^2
    # This is sum_j diff[i,j] * inv_dist_sq[i,j]
    # diff is (N, N, 3), inv_dist_sq is (N, N)
    # We want for each i: sum_j diff[i,j] * inv_dist_sq[i,j]
    # This is a matrix-vector product per row, or einsum
    grad = np.einsum('ijk,ij->ik', diff, inv_dist_sq)
    return grad

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _unconstrained_energy(theta_phi):
    """Energy as function of spherical coordinates (theta, phi).
    theta: polar angle (0 to pi)
    phi: azimuthal angle (0 to 2*pi)
    """
    theta = theta_phi[0:N]
    phi = theta_phi[N:2*N]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    pts = np.column_stack([x, y, z])
    return _log_energy(pts)

def _unconstrained_grad(theta_phi):
    """Gradient of energy w.r.t. (theta, phi)."""
    theta = theta_phi[0:N]
    phi = theta_phi[N:2*N]
    
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    pts = np.column_stack([x, y, z])
    
    # Gradient on sphere
    grad_pts = _grad_log_energy(pts)
    
    # Chain rule: dE/dtheta = grad_pts . dpts/dtheta
    # dpts/dtheta = (cos(theta)cos(phi), cos(theta)sin(phi), -sin(theta))
    dpts_dtheta = np.column_stack([
        np.cos(theta) * np.cos(phi),
        np.cos(theta) * np.sin(phi),
        -np.sin(theta)
    ])
    grad_theta = np.sum(grad_pts * dpts_dtheta, axis=1)
    
    # dpts/dphi = (-sin(theta)sin(phi), sin(theta)cos(phi), 0)
    dpts_dphi = np.column_stack([
        -np.sin(theta) * np.sin(phi),
        np.sin(theta) * np.cos(phi),
        np.zeros(N)
    ])
    grad_phi = np.sum(grad_pts * dpts_dphi, axis=1)
    
    return np.concatenate([grad_theta, grad_phi])

def _fibonacci_initialization():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts

def _pts_to_theta_phi(pts):
    """Convert points on sphere to (theta, phi) coordinates."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    return np.concatenate([theta, phi])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with Fibonacci spiral
    pts0 = _fibonacci_initialization()
    best_pts = pts0.copy()
    best_energy = _log_energy(pts0)
    
    # Try a few different initializations
    initializations = [pts0]
    
    # Jittered Fibonacci
    rng = np.random.default_rng(42)
    for seed in [1, 2, 3, 4, 5]:
        if time.time() > deadline:
            break
        rng = np.random.default_rng(seed)
        jitter = rng.normal(0, 0.05, (N, 3))
        pts_jitter = _project_to_sphere(pts0 + jitter)
        initializations.append(pts_jitter)
    
    # Random initialization
    for seed in [100, 101, 102]:
        if time.time() > deadline:
            break
        rng = np.random.default_rng(seed)
        pts_rand = rng.normal(0, 1, (N, 3))
        pts_rand = _project_to_sphere(pts_rand)
        initializations.append(pts_rand)
    
    for init_pts in initializations:
        if time.time() > deadline:
            break
        
        theta_phi0 = _pts_to_theta_phi(init_pts)
        
        try:
            result = minimize(
                _unconstrained_energy,
                theta_phi0,
                jac=_unconstrained_grad,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.success or result.fun < best_energy:
                theta_phi_opt = result.x
                theta = theta_phi_opt[0:N]
                phi = theta_phi_opt[N:2*N]
                x = np.sin(theta) * np.cos(phi)
                y = np.sin(theta) * np.sin(phi)
                z = np.cos(theta)
                pts_opt = np.column_stack([x, y, z])
                
                # Verify validity
                if np.all(np.isfinite(pts_opt)) and np.all(np.linalg.norm(pts_opt, axis=1) > 0):
                    energy = _log_energy(pts_opt)
                    if energy < best_energy:
                        best_energy = energy
                        best_pts = pts_opt
        except Exception:
            continue
    
    # Final high-precision polish
    if time.time() < deadline - 5.0:
        theta_phi0 = _pts_to_theta_phi(best_pts)
        try:
            result = minimize(
                _unconstrained_energy,
                theta_phi0,
                jac=_unconstrained_grad,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if result.fun < best_energy:
                theta_phi_opt = result.x
                theta = theta_phi_opt[0:N]
                phi = theta_phi_opt[N:2*N]
                x = np.sin(theta) * np.cos(phi)
                y = np.sin(theta) * np.sin(phi)
                z = np.cos(theta)
                pts_opt = np.column_stack([x, y, z])
                if np.all(np.isfinite(pts_opt)):
                    best_pts = pts_opt
        except Exception:
            pass
    
    # Ensure all points are on the unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    # Check for very close points
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    if min_dist < 1e-9:
        # Fallback to Fibonacci
        best_pts = _fibonacci_initialization()
    
    return best_pts
# EVOLVE-BLOCK-END
