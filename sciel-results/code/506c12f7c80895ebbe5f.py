"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
from scipy.optimize import basinhopping, minimize


def _cartesian_to_spherical(pts):
    """Convert Cartesian points (N, 3) to Spherical coordinates (N, 2).
    Returns theta (polar, 0..pi) and phi (azimuth, 0..2pi)."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    norms = np.linalg.norm(pts, axis=1)
    norms[norms < 1e-12] = 1.0
    theta = np.arccos(np.clip(z / norms, -1.0, 1.0))
    phi = np.arctan2(y, x)
    phi[phi < 0] += 2 * np.pi
    return np.column_stack([theta, phi])


def _spherical_to_cartesian(params):
    """Convert Spherical coordinates (N, 2) to Cartesian points (N, 3)."""
    theta = params[:, 0]
    phi = params[:, 1]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])


def _log_energy_cartesian(pts):
    """Compute logarithmic energy for Cartesian points."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    i_upper, j_upper = np.triu_indices(N, k=1)
    d = dists[i_upper, j_upper]
    d = np.maximum(d, 1e-12)
    terms = np.log(1.0 / d)
    return math.fsum(terms.tolist())


def _gradient_cartesian(pts):
    """Compute projected gradient of logarithmic energy for Cartesian points."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    dists_safe = np.where(dists < 1e-12, 1e-12, dists)
    
    W = np.zeros((N, N))
    mask = ~np.eye(N, dtype=bool)
    W[mask] = 1.0 / (dists_safe[mask] ** 2)
    
    sum_W = W.sum(axis=1, keepdims=True)
    grad = W @ pts - sum_W * pts
    
    dot_prod = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot_prod * pts
    return grad


def _objective_sph(params):
    """Objective function for Spherical coordinates."""
    pts = _spherical_to_cartesian(params)
    return _log_energy_cartesian(pts)


def _gradient_sph(params):
    """Gradient of the objective function with respect to Spherical coordinates."""
    pts = _spherical_to_cartesian(params)
    theta = params[:, 0]
    phi = params[:, 1]
    
    grad_cart = _gradient_cartesian(pts)
    
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    sin_p = np.sin(phi)
    cos_p = np.cos(phi)
    
    dx_dt = cos_t * cos_p
    dy_dt = cos_t * sin_p
    dz_dt = -sin_t
    
    dx_dp = -sin_t * sin_p
    dy_dp = sin_t * cos_p
    dz_dp = 0.0
    
    grad_theta = grad_cart[:, 0] * dx_dt + grad_cart[:, 1] * dy_dt + grad_cart[:, 2] * dz_dt
    grad_phi = grad_cart[:, 0] * dx_dp + grad_cart[:, 1] * dy_dp + grad_cart[:, 2] * dz_dp
    
    return np.column_stack([grad_theta, grad_phi])


def construct():
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts_cart = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    norms = np.linalg.norm(pts_cart, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    pts_cart = pts_cart / norms

    # Convert to Spherical coordinates
    params_sph = _cartesian_to_spherical(pts_cart)
    
    def fun(x):
        return _objective_sph(x)
    
    def jac(x):
        return _gradient_sph(x).flatten()
        
    # Basin Hopping parameters
    # L-BFGS-B is a good local minimizer for this smooth landscape
    # We use 50 iterations to stay well within the 120s time limit
    try:
        res = basinhopping(
            fun, 
            params_sph, 
            minimizer_kwargs={'method': 'L-BFGS-B', 'jac': jac, 'maxiter': 1000},
            niter=50,
            stepsize=0.5,
            T=1.0,
            disp=False
        )
        best_params = res.x
        best_pts = _spherical_to_cartesian(best_params)
    except Exception:
        best_pts = pts_cart

    # Final normalization and check
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    best_pts = best_pts / norms
    
    if not np.all(np.isfinite(best_pts)):
        return pts_cart
        
    return best_pts
# EVOLVE-BLOCK-END
