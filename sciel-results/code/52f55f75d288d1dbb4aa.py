"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize, Bounds

N = 100
DEADLINE = 115.0  # safety margin before 120s kill
START_TIME = time.time()


def _remaining():
    return DEADLINE - (time.time() - START_TIME)


def _energy_and_grad(pts):
    """Compute logarithmic energy and its gradient for points on the sphere.
    
    Energy: sum_{i<j} log(1 / |x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient w.r.t. x_i: sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    (This is the gradient of the energy in R^3; we project onto the sphere.)
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    
    # Pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist = np.linalg.norm(diff, axis=2)  # (n, n)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist, 1.0)
    
    # Energy: -sum_{i<j} log(dist_ij)
    # Use upper triangle only
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each i, grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # diff[i,j] = x_i - x_j, so grad_i = sum_j diff[i,j] / dist[i,j]^2
    inv_dist2 = 1.0 / (dist * dist)
    np.fill_diagonal(inv_dist2, 0.0)
    grad = np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)  # (n, 3)
    
    return energy, grad


def _project_to_sphere(pts):
    """Normalize each point to unit length."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using golden-angle spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    """Generate n random points on the unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _optimize_from_start(pts0, maxiter=200, tol=1e-10):
    """Run L-BFGS-B optimization on the sphere using a parameterization."""
    pts0 = np.asarray(pts0, dtype=np.float64)
    n = pts0.shape[0]
    
    # Use spherical coordinates as parameters: theta (polar), phi (azimuth)
    # theta in [0, pi], phi in [0, 2*pi)
    # x = sin(theta)*cos(phi), y = sin(theta)*sin(phi), z = cos(theta)
    
    # Extract initial spherical coords
    theta = np.arccos(np.clip(pts0[:, 2], -1.0, 1.0))
    phi = np.arctan2(pts0[:, 1], pts0[:, 0])
    params0 = np.concatenate([theta, phi])
    
    def obj(params):
        th = params[:n]
        ph = params[n:]
        x = np.sin(th) * np.cos(ph)
        y = np.sin(th) * np.sin(ph)
        z = np.cos(th)
        pts = np.column_stack([x, y, z])
        e, _ = _energy_and_grad(pts)
        return e
    
    def gradient(params):
        th = params[:n]
        ph = params[n:]
        x = np.sin(th) * np.cos(ph)
        y = np.sin(th) * np.sin(ph)
        z = np.cos(th)
        pts = np.column_stack([x, y, z])
        e, g = _energy_and_grad(pts)
        
        # Chain rule: dE/dtheta_i = sum_k g_i[k] * dx_k/dtheta_i
        # dx/dtheta = (cos(theta)*cos(phi), cos(theta)*sin(phi), -sin(theta))
        # dy/dtheta = ...
        # dz/dtheta = -sin(theta)
        st = np.sin(th)
        ct = np.cos(th)
        cp = np.cos(ph)
        sp = np.sin(ph)
        
        # dE/dtheta_i = g_i[0]*ct*cp + g_i[1]*ct*sp + g_i[2]*(-st)
        dE_dth = g[:, 0] * ct * cp + g[:, 1] * ct * sp - g[:, 2] * st
        
        # dE/dphi_i = g_i[0]*(-st*sp) + g_i[1]*(st*cp) + g_i[2]*0
        dE_dph = -g[:, 0] * st * sp + g[:, 1] * st * cp
        
        return np.concatenate([dE_dth, dE_dph])
    
    bounds = [(0, np.pi)] * n + [(0, 2 * np.pi)] * n
    
    result = minimize(obj, params0, jac=gradient, method='L-BFGS-B',
                      bounds=bounds, options={'maxiter': maxiter, 'ftol': tol, 'gtol': 1e-8})
    
    th = result.x[:n]
    ph = result.x[n:]
    x = np.sin(th) * np.cos(ph)
    y = np.sin(th) * np.sin(ph)
    z = np.cos(th)
    pts = np.column_stack([x, y, z])
    
    return pts, result.fun


def _polish(pts, maxiter=500, tol=1e-12):
    """High-precision polish pass with tighter tolerances."""
    pts = _optimize_from_start(pts, maxiter=maxiter, tol=tol)[0]
    return pts


def construct():
    global START_TIME
    START_TIME = time.time()
    
    rng = np.random.default_rng(42)
    
    # Start with Fibonacci spiral
    pts = _fibonacci_sphere(N)
    best_pts = pts.copy()
    best_energy = _energy_and_grad(pts)[0]
    
    # First optimization pass
    if _remaining() > 10:
        pts, e = _optimize_from_start(pts, maxiter=300, tol=1e-10)
        if e < best_energy:
            best_pts = pts.copy()
            best_energy = e
    
    # Try a few random restarts if time permits
    n_restarts = 5
    for r in range(n_restarts):
        if _remaining() < 15:
            break
        pts_rand = _random_sphere(N, rng)
        pts_opt, e = _optimize_from_start(pts_rand, maxiter=200, tol=1e-10)
        if e < best_energy:
            best_pts = pts_opt.copy()
            best_energy = e
    
    # Polish the best solution
    if _remaining() > 10:
        polished = _polish(best_pts, maxiter=500, tol=1e-12)
        e_pol = _energy_and_grad(polished)[0]
        if e_pol < best_energy:
            best_pts = polished.copy()
            best_energy = e_pol
    
    # Final projection to ensure unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    return best_pts
