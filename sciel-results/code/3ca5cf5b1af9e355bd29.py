"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _log_energy(pts):
    """Compute log(1/d) energy sum over pairs, with numerical safety."""
    n = len(pts)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Zero out diagonal to avoid log(0)
    np.fill_diagonal(dists, 1.0)
    # Clip to avoid log of very small numbers causing issues
    dists = np.maximum(dists, 1e-15)
    # Sum over upper triangle (i < j)
    iu = np.triu_indices(n, k=1)
    vals = -np.log(dists[iu])
    return float(math.fsum(vals))

def _log_energy_grad(pts):
    """Compute gradient of log energy w.r.t. point positions (not projected)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    dists = np.maximum(dists, 1e-15)
    
    # Gradient of -log(d_ij) w.r.t. x_i is diff_ij / d_ij^2
    # For each pair (i,j), contribution to grad[i] is (x_i - x_j) / d_ij^2
    # contribution to grad[j] is (x_j - x_i) / d_ij^2 = -diff_ij / d_ij^2
    
    # diff[i,j] = x_i - x_j
    # grad[i] += sum_j (x_i - x_j) / d_ij^2
    # grad[j] -= sum_i (x_i - x_j) / d_ij^2
    
    inv_d2 = 1.0 / (dists ** 2)
    inv_d2[np.diag_indices(n)] = 0.0
    
    # grad[i] = sum_j diff[i,j] * inv_d2[i,j]
    grad = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    return grad

def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _spherical_energy_and_grad(theta_phi):
    """
    theta_phi: (n, 2) array of (theta, phi) angles
    Returns (energy, gradient in (theta, phi) space)
    """
    theta = theta_phi[:, 0]
    phi = theta_phi[:, 1]
    
    # Convert to Cartesian
    sin_t = np.sin(theta)
    x = sin_t * np.cos(phi)
    y = sin_t * np.sin(phi)
    z = np.cos(theta)
    pts = np.column_stack([x, y, z])
    
    energy = _log_energy(pts)
    grad_cart = _log_energy_grad(pts)
    
    # Chain rule: grad_theta, grad_phi
    # dx/dtheta = cos(theta)cos(phi), dy/dtheta = cos(theta)sin(phi), dz/dtheta = -sin(theta)
    # dx/dphi = -sin(theta)sin(phi), dy/dphi = sin(theta)cos(phi), dz/dphi = 0
    
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    
    grad_theta = grad_cart[:, 0] * ct * cp + grad_cart[:, 1] * ct * sp + grad_cart[:, 2] * (-st)
    grad_phi = grad_cart[:, 0] * (-st * sp) + grad_cart[:, 1] * (st * cp) + grad_cart[:, 2] * 0.0
    
    grad = np.column_stack([grad_theta, grad_phi])
    return energy, grad

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Fibonacci spiral start
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Convert to spherical coordinates
    theta = np.arccos(np.clip(pts[:, 2], -1.0, 1.0))
    phi_s = np.arctan2(pts[:, 1], pts[:, 0])
    tp = np.column_stack([theta, phi_s])
    
    best_energy = _spherical_energy_and_grad(tp)[0]
    best_tp = tp.copy()
    
    # Use L-BFGS-B to minimize
    from scipy.optimize import minimize
    
    def objective(tp_flat):
        tp_2d = tp_flat.reshape(-1, 2)
        return _spherical_energy_and_grad(tp_2d)[0]
    
    def gradient(tp_flat):
        tp_2d = tp_flat.reshape(-1, 2)
        return _spherical_energy_and_grad(tp_2d)[1].ravel()
    
    # Try L-BFGS-B with time limit
    result = minimize(
        objective,
        tp.ravel(),
        jac=gradient,
        method='L-BFGS-B',
        options={'maxiter': 1000, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    if result.fun < best_energy:
        best_energy = result.fun
        best_tp = result.x.reshape(-1, 2)
    
    # Check time and try a second round if we have time
    if time.time() < deadline - 5.0:
        # Perturb and try again
        rng = np.random.default_rng(42)
        for restart in range(3):
            if time.time() > deadline - 5.0:
                break
            # Small random perturbation
            perturbed = best_tp + rng.normal(0, 0.1, size=best_tp.shape)
            result2 = minimize(
                objective,
                perturbed.ravel(),
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result2.fun < best_energy:
                best_energy = result2.fun
                best_tp = result2.x.reshape(-1, 2)
    
    # Convert back to Cartesian
    theta = best_tp[:, 0]
    phi = best_tp[:, 1]
    sin_t = np.sin(theta)
    x = sin_t * np.cos(phi)
    y = sin_t * np.sin(phi)
    z = np.cos(theta)
    pts = np.column_stack([x, y, z])
    
    # Final normalization
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    return pts
# EVOLVE-BLOCK-END
