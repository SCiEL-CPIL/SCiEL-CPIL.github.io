import numpy as np
from scipy.optimize import minimize
import time

N = 150

def energy_grad_cartesian(pts):
    """
    Calculate energy and gradient for points on the unit sphere.
    pts: (N, 3) array.
    Returns: energy (float), grad (N, 3) array.
    """
    # pts is assumed to be on the unit sphere, but we don't enforce it in calc
    # diff[i, j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    
    # dist[i, j] = || pts[i] - pts[j] ||
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    
    # Energy: sum_{i<j} log(1/d_ij) = -0.5 * sum_{i!=j} log(d_ij)
    # To avoid log(0), we only sum over mask
    log_dist = np.zeros_like(dist)
    log_dist[mask] = np.log(dist[mask])
    energy = -0.5 * np.sum(log_dist)
    
    # Gradient
    # g_i = sum_{j!=i} (pts[j] - pts[i]) / dist[i,j]^2
    # Note: diff[i,j] = pts[i] - pts[j]
    # So pts[j] - pts[i] = -diff[i,j]
    
    dist_sq = dist**2
    dist_sq[~mask] = 1.0 # Avoid div by zero
    
    # weights[i, j] = 1 / dist[i,j]^2 for i!=j, 0 otherwise
    weights = np.zeros_like(dist_sq)
    weights[mask] = 1.0 / dist_sq[mask]
    
    # grad[i] = sum_j weights[i, j] * (-diff[i, j])
    # = - sum_j weights[i, j] * diff[i, j]
    grad = -np.sum(weights[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad

def energy_grad_spherical(theta, phi):
    """
    Calculate energy and gradient in spherical coordinates.
    theta: (N,), phi: (N,)
    """
    pts = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])
    
    energy, grad_cart = energy_grad_cartesian(pts)
    
    # Basis vectors for spherical coords
    # d(pts)/d(theta)
    d_pts_d_theta = np.column_stack([
        np.cos(theta) * np.cos(phi),
        np.cos(theta) * np.sin(phi),
        -np.sin(theta)
    ])
    
    # d(pts)/d(phi)
    d_pts_d_phi = np.column_stack([
        -np.sin(theta) * np.sin(phi),
        np.sin(theta) * np.cos(phi),
        np.zeros(N)
    ])
    
    # Project Cartesian gradient onto spherical basis
    grad_theta = np.sum(grad_cart * d_pts_d_theta, axis=1)
    grad_phi = np.sum(grad_cart * d_pts_d_phi, axis=1)
    
    grad_flat = np.concatenate([grad_theta, grad_phi])
    
    return energy, grad_flat

def polish(pts, deadline):
    """
    Riemannian Gradient Descent polish.
    """
    pts = pts.copy()
    lr = 1e-3
    prev_e = np.inf
    
    for k in range(2000):
        if time.time() > deadline:
            break
            
        e, g = energy_grad_cartesian(pts)
        
        # Project gradient to tangent space
        g_dot_p = np.sum(g * pts, axis=1, keepdims=True)
        g_tan = g - g_dot_p * pts
        
        # Check convergence
        g_norm = np.linalg.norm(g_tan)
        if g_norm < 1e-10:
            break
            
        # Update
        new_pts = pts - lr * g_tan
        # Project to sphere
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        new_pts = new_pts / norms
        
        # Evaluate new energy
        new_e, _ = energy_grad_cartesian(new_pts)
        
        if new_e < e:
            pts = new_pts
            # Optional: increase lr slightly if progress is good
            # lr = min(lr * 1.05, 1e-2)
        else:
            # Backtrack
            lr *= 0.5
            if lr < 1e-8:
                break
                
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    candidates = []
    
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    theta = np.arccos(z)
    candidates.append((theta, phi))
    
    # 2. Random Uniform
    np.random.seed(42)
    theta_rand = np.random.uniform(1e-6, np.pi - 1e-6, N)
    phi_rand = np.random.uniform(0, 2*np.pi, N)
    candidates.append((theta_rand, phi_rand))
    
    # 3. Fibonacci with Noise
    theta_noise = theta + np.random.normal(0, 0.2, N)
    phi_noise = phi + np.random.normal(0, 0.2, N)
    theta_noise = np.clip(theta_noise, 1e-6, np.pi - 1e-6)
    candidates.append((theta_noise, phi_noise))
    
    best_energy = np.inf
    best_pts = None
    
    for idx, (th, ph) in enumerate(candidates):
        if time.time() > deadline:
            break
            
        x0 = np.concatenate([th, ph])
        
        try:
            res = minimize(
                fun=energy_grad_spherical,
                x0=x0,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-6, np.pi - 1e-6)] * N + [(0, 2*np.pi)] * N,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if res.success:
                th_opt, ph_opt = res.x[:N], res.x[N:]
                pts = np.column_stack([
                    np.sin(th_opt) * np.cos(ph_opt),
                    np.sin(th_opt) * np.sin(ph_opt),
                    np.cos(th_opt)
                ])
                
                # Polish
                pts = polish(pts, deadline)
                
                e, _ = energy_grad_cartesian(pts)
                if e < best_energy:
                    best_energy = e
                    best_pts = pts
                    
        except Exception as e:
            pass
            
    if best_pts is None:
        # Fallback to initial Fibonacci
        pts = np.column_stack([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        best_pts = pts
        
    return best_pts

# EVOLVE-BLOCK-END
