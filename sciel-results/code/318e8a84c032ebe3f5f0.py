"""Rewritten program: L-BFGS-B on spherical coordinates for log-energy minimization."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START

def _points_to_spherical(pts):
    """Convert Cartesian points to spherical coordinates (theta, phi)."""
    # theta is polar angle [0, pi], phi is azimuthal angle [0, 2pi]
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    r = np.maximum(r, 1e-15)
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.arctan2(y, x)
    # Ensure phi is in [0, 2pi]
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    return theta, phi

def _spherical_to_points(theta, phi):
    """Convert spherical coordinates to Cartesian points on unit sphere."""
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])

def _energy_and_grad_spherical(params):
    """
    Compute logarithmic energy and gradient with respect to spherical coordinates.
    params: array of shape (2N,). First N are theta, next N are phi.
    """
    N_local = len(params) // 2
    theta = params[:N_local]
    phi = params[N_local:]
    
    pts = _spherical_to_points(theta, phi)
    
    # Compute squared distances
    # d^2 = 2 - 2 * dot(xi, xj)
    dots = pts @ pts.T
    sq_dists = 2.0 - 2.0 * dots
    np.fill_diagonal(sq_dists, 1.0)
    # Clip for numerical stability
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    # Energy: sum_{i<j} log(1/d) = -0.5 * sum_{i<j} log(d^2)
    iu = np.triu_indices(N_local, k=1)
    log_dists_sq = np.log(sq_dists[iu])
    energy = -0.5 * math.fsum(log_dists_sq.tolist())
    
    # Gradient in Cartesian space
    # grad_xi = sum_{j != i} (xj - xi) / d_ij^2
    # weights[i, j] = 1 / d_ij^2
    weights = 1.0 / sq_dists
    np.fill_diagonal(weights, 0.0)
    
    # diffs[i, j] = xi - xj
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    # grad_cart[i] = sum_j diffs[i, j] * weights[i, j]
    # Note: sum_j (xi - xj)/d^2 is the force on i from j? 
    # d/dxi log(1/|xi-xj|) = -1/|xi-xj| * (xi-xj)/|xi-xj| = -(xi-xj)/|xi-xj|^2
    # So grad = - sum_j (xi-xj)/d^2 = sum_j (xj-xi)/d^2
    # My diffs is xi - xj. So grad_cart = - sum_j diffs[i,j] * weights[i,j]
    grad_cart = -np.sum(diffs * weights[..., np.newaxis], axis=1)
    
    # Project Cartesian gradient onto spherical basis vectors
    # dxi/dtheta = (cos(theta)cos(phi), cos(theta)sin(phi), -sin(theta))
    # dxi/dphi = (-sin(theta)sin(phi), sin(theta)cos(phi), 0)
    
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    
    # Basis vectors
    # shape (N, 3)
    basis_theta = np.column_stack([
        cos_theta * cos_phi,
        cos_theta * sin_phi,
        -sin_theta
    ])
    basis_phi = np.column_stack([
        -sin_theta * sin_phi,
        sin_theta * cos_phi,
        np.zeros(N_local)
    ])
    
    # Gradient in spherical coords
    # grad_theta[i] = dot(grad_cart[i], basis_theta[i])
    grad_theta = np.sum(grad_cart * basis_theta, axis=1)
    grad_phi = np.sum(grad_cart * basis_phi, axis=1)
    
    # Concatenate
    grad = np.concatenate([grad_theta, grad_phi])
    
    return energy, grad

def _optimize_spherical(theta, phi, max_iter=500, time_limit=None, start_time=None):
    """
    Optimize using L-BFGS-B on spherical coordinates.
    """
    if start_time is None:
        start_time = time.time()
        
    params = np.concatenate([theta, phi])
    
    # Bounds: theta in [1e-6, pi - 1e-6], phi in [0, 2pi]
    # Using small epsilon to avoid poles where phi is undefined
    eps = 1e-6
    bounds = [(eps, np.pi - eps)] * N + [(0, 2 * np.pi)] * N
    
    def obj_and_grad(p):
        if time_limit is not None and time.time() - start_time > time_limit:
            # Return a large value to signal termination? 
            # L-BFGS-B doesn't support early exit easily via callback in all versions.
            # We rely on max_iter being small enough or checking time outside.
            pass
        e, g = _energy_and_grad_spherical(p)
        return e, g
        
    # L-BFGS-B
    res = minimize(
        obj_and_grad,
        params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
    )
    
    return res.x, res.fun

def _initial_fibonacci(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts

def _initial_random(n, seed=42):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def construct():
    deadline = time.time() + 145  # Leave 5s margin for 150s limit
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Fibonacci start
    pts = _initial_fibonacci(N)
    theta, phi = _points_to_spherical(pts)
    
    t0 = time.time()
    time_left = deadline - t0 - 5
    if time_left > 1:
        try:
            x_opt, e_opt = _optimize_spherical(theta, phi, max_iter=200, time_limit=time_left, start_time=t0)
            if e_opt < best_energy:
                best_energy = e_opt
                # Convert back to Cartesian
                theta_opt = x_opt[:N]
                phi_opt = x_opt[N:]
                best_pts = _spherical_to_points(theta_opt, phi_opt)
        except Exception:
            pass

    # Phase 2: Random restarts
    # Use remaining time for random restarts
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        # Estimate time per restart. L-BFGS-B is fast. ~0.1-0.5s per run for 150 pts.
        # Let's try to fit as many as possible within the time budget.
        # We'll do a loop that checks time.
        num_restarts = 0
        while time.time() < deadline - 10:
            seed = 42 + num_restarts * 1000
            pts_r = _initial_random(N, seed=seed)
            theta_r, phi_r = _points_to_spherical(pts_r)
            
            t0 = time.time()
            time_left = deadline - t0 - 5
            if time_left < 1:
                break
            
            try:
                x_opt, e_opt = _optimize_spherical(theta_r, phi_r, max_iter=100, time_limit=time_left, start_time=t0)
                if e_opt < best_energy:
                    best_energy = e_opt
                    theta_opt = x_opt[:N]
                    phi_opt = x_opt[N:]
                    best_pts = _spherical_to_points(theta_opt, phi_opt)
            except Exception:
                pass
            num_restarts += 1
            
            # Safety break if we've done a lot
            if num_restarts > 50: 
                break

    # Phase 3: Polish best solution if time remains
    if best_pts is not None and time.time() < deadline - 5:
        t0 = time.time()
        time_left = deadline - t0 - 2
        if time_left > 1:
            # Convert best_pts to spherical
            theta_best, phi_best = _points_to_spherical(best_pts)
            try:
                x_opt, e_opt = _optimize_spherical(theta_best, phi_best, max_iter=500, time_limit=time_left, start_time=t0)
                if e_opt < best_energy:
                    best_energy = e_opt
                    theta_opt = x_opt[:N]
                    phi_opt = x_opt[N:]
                    best_pts = _spherical_to_points(theta_opt, phi_opt)
            except Exception:
                pass

    # Final safety check: ensure points are valid
    if best_pts is None:
        # Fallback to Fibonacci if everything failed
        best_pts = _initial_fibonacci(N)
        
    return _project_to_sphere(best_pts)

# EVOLVE-BLOCK-END
