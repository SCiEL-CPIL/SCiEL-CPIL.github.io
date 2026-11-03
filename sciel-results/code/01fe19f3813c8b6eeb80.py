import numpy as np
from scipy.optimize import minimize
import time
import math

N = 100

def _energy_and_grad(params):
    # params: (N, 2) theta, phi
    theta = params[:, 0]
    phi = params[:, 1]
    
    # Convert to Cartesian
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    pts = np.column_stack((x, y, z))
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Mask diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    # Regularized distance squared
    eps = 1e-6
    dist_sq_reg = dist_sq + eps**2
    
    log_dist_sq = np.log(dist_sq_reg)
    np.fill_diagonal(log_dist_sq, 0.0)
    
    energy = -0.5 * np.sum(log_dist_sq)
    
    coeff = diff / dist_sq_reg[:, :, np.newaxis]
    coeff[np.arange(N), np.arange(N), :] = 0.0
    
    grad_x = -np.sum(coeff, axis=1)
    
    d_pts_d_theta = np.column_stack((
        np.cos(theta) * np.cos(phi),
        np.cos(theta) * np.sin(phi),
        -np.sin(theta)
    ))
    
    d_pts_d_phi = np.column_stack((
        -np.sin(theta) * np.sin(phi),
        np.sin(theta) * np.cos(phi),
        np.zeros(N)
    ))
    
    grad_theta = np.sum(grad_x * d_pts_d_theta, axis=1)
    grad_phi = np.sum(grad_x * d_pts_d_phi, axis=1)
    
    grad = np.column_stack((grad_theta, grad_phi))
    
    return energy, grad


def _sphere_energy_grad(pts):
    """Compute energy and gradient directly on sphere points (N,3)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    
    eps = 1e-8
    dist_sq_reg = dist_sq + eps**2
    
    log_dist_sq = np.log(dist_sq_reg)
    np.fill_diagonal(log_dist_sq, 0.0)
    
    energy = -0.5 * np.sum(log_dist_sq)
    
    coeff = diff / dist_sq_reg[:, :, np.newaxis]
    coeff[np.arange(N), np.arange(N), :] = 0.0
    
    grad = -np.sum(coeff, axis=1)
    
    return energy, grad


def _exp_map(pts, tangent):
    """Exponential map on the unit sphere: move pts along tangent vectors."""
    # tangent should be orthogonal to pts
    # Remove radial component
    radial = np.sum(tangent * pts, axis=1, keepdims=True)
    tangent = tangent - radial * pts
    
    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    # Avoid division by zero
    safe_norms = np.where(norms < 1e-15, 1.0, norms)
    
    # For small angles, use series expansion
    angles = np.linalg.norm(tangent, axis=1)
    sin_angles = np.where(angles < 1e-10, angles, np.sin(angles) / safe_norms[:, 0])
    cos_angles = np.where(angles < 1e-10, 1.0 - 0.5 * angles**2, np.cos(angles))
    
    new_pts = cos_angles[:, np.newaxis] * pts + sin_angles[:, np.newaxis] * tangent
    
    # Normalize to ensure unit sphere
    new_norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    new_pts = new_pts / new_norms
    
    return new_pts


def _log_map(pts, ref_pts):
    """Logarithmic map: compute tangent vector from ref_pts to pts."""
    # ref_pts should be the base point
    cos_angle = np.sum(pts * ref_pts, axis=1, keepdims=True)
    cos_angle = np.clip(cos_angle, -1.0 + 1e-15, 1.0 - 1e-15)
    angles = np.arccos(cos_angle[:, 0])
    
    sin_angles = np.where(angles < 1e-10, angles, np.sin(angles) / angles[:, 0])
    
    tangent = sin_angles * (pts - cos_angle * ref_pts)
    
    return tangent


def _riemannian_polish(pts, max_iter=500, tol=1e-14):
    """Polish using Riemannian gradient descent with backtracking line search."""
    pts = pts.copy()
    
    for it in range(max_iter):
        energy, grad = _sphere_energy_grad(pts)
        
        # Check convergence
        if np.max(np.abs(grad)) < tol:
            break
        
        # Project gradient to tangent space
        radial = np.sum(grad * pts, axis=1, keepdims=True)
        tangent_grad = grad - radial * pts
        
        # Backtracking line search
        step = 0.01
        for ls_iter in range(30):
            new_pts = _exp_map(pts, -step * tangent_grad)
            new_energy, _ = _sphere_energy_grad(new_pts)
            
            if new_energy < energy - 1e-4 * step * np.sum(tangent_grad * (-tangent_grad)):
                pts = new_pts
                break
            step *= 0.5
        
        else:
            break
    
    return pts


def _precise_energy(pts):
    """Compute energy using math.fsum for maximum precision."""
    total = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return float('inf')
            total += math.log(1.0 / d)
    return math.fsum([math.log(1.0 / np.linalg.norm(pts[i] - pts[j])) 
                      for i in range(N) for j in range(i + 1, N)])

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s buffer
    
    best_energy = float('inf')
    best_params = None
    
    def _params_to_pts(params):
        theta = params[:, 0]
        phi = params[:, 1]
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        pts = np.column_stack((x, y, z))
        norms = np.linalg.norm(pts, axis=1)
        return pts / norms[:, np.newaxis]
    
    # Initial guesses
    # 1. Fibonacci
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    theta_fib = np.arccos(z)
    phi_fib = np.pi * (1.0 + np.sqrt(5.0)) * i
    init_fib = np.column_stack((theta_fib, phi_fib))
    
    # 2. Random
    np.random.seed(42)
    init_rand = np.random.uniform(0, np.pi, (N, 1))
    init_rand = np.hstack((init_rand, np.random.uniform(0, 2*np.pi, (N, 1))))
    
    inits = [init_fib, init_rand]
    
    for init in inits:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(_energy_and_grad, init, method='L-BFGS-B', jac=True, 
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if res.success or res.fun < best_energy:
                best_energy = res.fun
                best_params = res.x
        except Exception as e:
            pass
            
    # If we have time, try more random restarts
    while time.time() < deadline - 30.0:
        init_rand = np.random.uniform(0, np.pi, (N, 1))
        init_rand = np.hstack((init_rand, np.random.uniform(0, 2*np.pi, (N, 1))))
        
        try:
            res = minimize(_energy_and_grad, init_rand, method='L-BFGS-B', jac=True, 
                           options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if res.fun < best_energy:
                best_energy = res.fun
                best_params = res.x
        except Exception:
            pass
            
    if best_params is None:
        best_params = init_fib
    
    # Convert back to Cartesian
    pts = _params_to_pts(best_params)
    
    # Riemannian polish pass on the sphere
    if time.time() < deadline - 5.0:
        pts = _riemannian_polish(pts, max_iter=1000, tol=1e-14)
    
    # Final high-precision polish with very small steps
    if time.time() < deadline - 2.0:
        for _ in range(50):
            energy, grad = _sphere_energy_grad(pts)
            radial = np.sum(grad * pts, axis=1, keepdims=True)
            tangent_grad = grad - radial * pts
            # Very small step
            pts = _exp_map(pts, -1e-8 * tangent_grad)
    
    # Normalize to ensure unit sphere
    norms = np.linalg.norm(pts, axis=1)
    pts = pts / norms[:, np.newaxis]
    
    return pts
