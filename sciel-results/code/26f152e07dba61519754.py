import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _energy_and_grad(theta_phi):
    theta = theta_phi[:, 0]
    phi = theta_phi[:, 1]
    
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    
    x = sin_theta * cos_phi
    y = sin_theta * sin_phi
    z = cos_theta
    points = np.column_stack((x, y, z))
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    d_sq = dist_sq[i_idx, j_idx]
    
    d_sq_clipped = np.maximum(d_sq, 1e-18)
    d = np.sqrt(d_sq_clipped)
    
    energy = -np.sum(np.log(d))
    
    inv_d_sq = np.zeros_like(dist_sq)
    np.divide(1.0, dist_sq, out=inv_d_sq, where=dist_sq > 1e-18)
    np.fill_diagonal(inv_d_sq, 0.0)
    
    # grad_cart[i] = sum_{j != i} (x_i - x_j) / d_ij^2
    # This is -dE/dx_i
    sum_term = np.sum(diff * inv_d_sq[:, :, np.newaxis], axis=1)
    grad_cart = -sum_term
    
    d_x_d_theta = np.column_stack((cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta))
    d_x_d_phi = np.column_stack((-sin_theta * sin_phi, sin_theta * cos_phi, np.zeros(N)))
    
    grad_theta = np.sum(grad_cart * d_x_d_theta, axis=1)
    grad_phi = np.sum(grad_cart * d_x_d_phi, axis=1)
    
    grad = np.column_stack((grad_theta, grad_phi))
    
    return energy, grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0_cart = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    theta0 = np.arccos(np.clip(z, -1.0, 1.0))
    phi0 = np.arctan2(x0_cart[:, 1], x0_cart[:, 0])
    
    x0 = np.column_stack((theta0, phi0))
    
    res = minimize(_energy_and_grad, x0, method='L-BFGS-B', jac=True)
    best_x = res.x
    best_energy = res.fun
    
    rng = np.random.RandomState(42)
    step_size = 0.5
    
    for _ in range(100):
        if time.time() > deadline:
            break
        noise = rng.normal(0, step_size, size=best_x.shape)
        x_perturbed = best_x + noise
        
        try:
            res = minimize(_energy_and_grad, x_perturbed, method='L-BFGS-B', jac=True)
            if res.fun < best_energy:
                best_energy = res.fun
                best_x = res.x
        except Exception:
            continue
            
    if time.time() < deadline - 5.0:
        try:
            res = minimize(_energy_and_grad, best_x, method='L-BFGS-B', jac=True, options={'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_energy:
                best_energy = res.fun
                best_x = res.x
        except Exception:
            pass
            
    theta = best_x[:, 0]
    phi = best_x[:, 1]
    
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    
    x = sin_theta * cos_phi
    y = sin_theta * sin_phi
    z = cos_theta
    
    points = np.column_stack((x, y, z))
    
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, np.newaxis]
    
    return points
