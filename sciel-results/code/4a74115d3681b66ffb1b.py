import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150

def _stereo_to_sphere(U):
    """
    Convert stereographic coordinates (u, v) to sphere points.
    Projection from North Pole (0,0,1).
    p = (2u / (1+u^2+v^2), 2v / (1+u^2+v^2), (1-u^2-v^2) / (1+u^2+v^2))
    """
    u, v = U[:, 0], U[:, 1]
    r2 = u**2 + v**2
    denom = 1.0 + r2
    px = 2.0 * u / denom
    py = 2.0 * v / denom
    pz = (1.0 - r2) / denom
    return np.column_stack([px, py, pz])

def _stereo_to_sphere_south(U):
    """
    Convert stereographic coordinates (u, v) to sphere points.
    Projection from South Pole (0,0,-1).
    p = (2u / (1+u^2+v^2), 2v / (1+u^2+v^2), (u^2+v^2-1) / (1+u^2+v^2))
    """
    u, v = U[:, 0], U[:, 1]
    r2 = u**2 + v**2
    denom = 1.0 + r2
    px = 2.0 * u / denom
    py = 2.0 * v / denom
    pz = (r2 - 1.0) / denom
    return np.column_stack([px, py, pz])

def _sphere_to_stereo(P):
    """
    Convert sphere points to stereographic coordinates (north pole projection).
    u = x / (1 - z), v = y / (1 - z)
    """
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    denom = 1.0 - z
    u = x / denom
    v = y / denom
    return np.column_stack([u, v])

def _sphere_to_stereo_south(P):
    """
    Convert sphere points to stereographic coordinates (south pole projection).
    u = x / (1 + z), v = y / (1 + z)
    """
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    denom = 1.0 + z
    u = x / denom
    v = y / denom
    return np.column_stack([u, v])

def _energy_and_grad_stereo(U):
    """
    Compute energy and gradient with respect to stereographic coordinates U.
    U: (N, 2)
    """
    P = _stereo_to_sphere(U)
    n = len(P)
    
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    
    log_dists_sq = np.log(dists_sq)
    np.fill_diagonal(log_dists_sq, 0.0)
    energy = -0.25 * np.sum(log_dists_sq)
    
    inv_dists_sq = 1.0 / dists_sq
    np.fill_diagonal(inv_dists_sq, 0.0)
    
    weighted_sum_P = inv_dists_sq @ P
    sum_inv_dists_sq = np.sum(inv_dists_sq, axis=1)
    grad_P = weighted_sum_P - P * sum_inv_dists_sq[:, np.newaxis]
    
    u, v = U[:, 0], U[:, 1]
    r2 = u**2 + v**2
    denom = (1.0 + r2)**2
    
    J_x_u = 2.0 * (1.0 - v**2) / denom
    J_x_v = -4.0 * u * v / denom
    J_y_u = -4.0 * u * v / denom
    J_y_v = 2.0 * (1.0 - u**2) / denom
    J_z_u = -4.0 * u / denom
    J_z_v = -4.0 * v / denom
    
    grad_U = np.zeros_like(U)
    grad_U[:, 0] = J_x_u * grad_P[:, 0] + J_y_u * grad_P[:, 1] + J_z_u * grad_P[:, 2]
    grad_U[:, 1] = J_x_v * grad_P[:, 0] + J_y_v * grad_P[:, 1] + J_z_v * grad_P[:, 2]
    
    return energy, grad_U

def _energy_and_grad_stereo_south(U):
    """
    Compute energy and gradient with respect to south-pole stereographic coordinates U.
    U: (N, 2)
    """
    P = _stereo_to_sphere_south(U)
    n = len(P)
    
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    
    log_dists_sq = np.log(dists_sq)
    np.fill_diagonal(log_dists_sq, 0.0)
    energy = -0.25 * np.sum(log_dists_sq)
    
    inv_dists_sq = 1.0 / dists_sq
    np.fill_diagonal(inv_dists_sq, 0.0)
    
    weighted_sum_P = inv_dists_sq @ P
    sum_inv_dists_sq = np.sum(inv_dists_sq, axis=1)
    grad_P = weighted_sum_P - P * sum_inv_dists_sq[:, np.newaxis]
    
    u, v = U[:, 0], U[:, 1]
    r2 = u**2 + v**2
    denom = (1.0 + r2)**2
    
    # Jacobian for south-pole stereographic projection
    # p_x = 2u / (1+r2), p_y = 2v / (1+r2), p_z = (r2-1) / (1+r2)
    # d(p_x)/du = 2(1 - v^2) / (1+r2)^2
    # d(p_x)/dv = -4uv / (1+r2)^2
    # d(p_y)/du = -4uv / (1+r2)^2
    # d(p_y)/dv = 2(1 - u^2) / (1+r2)^2
    # d(p_z)/du = 4u / (1+r2)^2
    # d(p_z)/dv = 4v / (1+r2)^2
    
    J_x_u = 2.0 * (1.0 - v**2) / denom
    J_x_v = -4.0 * u * v / denom
    J_y_u = -4.0 * u * v / denom
    J_y_v = 2.0 * (1.0 - u**2) / denom
    J_z_u = 4.0 * u / denom
    J_z_v = 4.0 * v / denom
    
    grad_U = np.zeros_like(U)
    grad_U[:, 0] = J_x_u * grad_P[:, 0] + J_y_u * grad_P[:, 1] + J_z_u * grad_P[:, 2]
    grad_U[:, 1] = J_x_v * grad_P[:, 0] + J_y_v * grad_P[:, 1] + J_z_v * grad_P[:, 2]
    
    return energy, grad_U

def _compute_energy(P):
    """Compute logarithmic energy from sphere points."""
    n = len(P)
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    log_dists_sq = np.log(dists_sq)
    np.fill_diagonal(log_dists_sq, 0.0)
    return -0.25 * np.sum(log_dists_sq)

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _halton_sphere(n, dims=3):
    """Generate points on sphere using Halton sequences."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = []
    for i in range(1, n + 1):
        u = halton(i, 2)
        v = halton(i, 3)
        w = halton(i, 5)
        # Map to sphere using spherical coordinates
        theta = 2 * np.pi * u
        phi = np.arccos(2 * v - 1)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        points.append([x, y, z])
    pts = np.array(points)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initializations in Sphere coords
    inits_sphere = []
    inits_sphere.append(_fibonacci_sphere(N))
    inits_sphere.append(_halton_sphere(N))
    
    # Random inits
    for s in range(15):
        rng = np.random.default_rng(s)
        pts = rng.standard_normal((N, 3))
        pts = _normalize(pts)
        inits_sphere.append(pts)
        
    # Perturbed Fibonacci
    for s in range(15):
        rng = np.random.default_rng(s + 100)
        scale = 0.05 + 0.03 * (s % 5)
        pts = _fibonacci_sphere(N) + rng.standard_normal((N, 3)) * scale
        pts = _normalize(pts)
        inits_sphere.append(pts)
        
    best_pts = None
    best_energy = float('inf')
    
    # Convert inits to both stereo charts and optimize
    for pts in inits_sphere:
        if time.time() > deadline - 15:
            break
        
        # Try north-pole stereographic
        try:
            if not np.any(pts[:, 2] > 0.999):
                U = _sphere_to_stereo(pts)
                if not np.any(~np.isfinite(U)):
                    res = minimize(_energy_and_grad_stereo, U, method='L-BFGS-B', jac=True, 
                                   options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-8})
                    U_opt = res.x
                    P_opt = _stereo_to_sphere(U_opt)
                    P_opt = _normalize(P_opt)
                    energy = _compute_energy(P_opt)
                    if energy < best_energy:
                        best_energy = energy
                        best_pts = P_opt.copy()
                        best_U = U_opt.copy()
                        best_chart = 'north'
        except Exception:
            pass
        
        # Try south-pole stereographic
        try:
            if not np.any(pts[:, 2] < -0.999):
                U = _sphere_to_stereo_south(pts)
                if not np.any(~np.isfinite(U)):
                    res = minimize(_energy_and_grad_stereo_south, U, method='L-BFGS-B', jac=True, 
                                   options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-8})
                    U_opt = res.x
                    P_opt = _stereo_to_sphere_south(U_opt)
                    P_opt = _normalize(P_opt)
                    energy = _compute_energy(P_opt)
                    if energy < best_energy:
                        best_energy = energy
                        best_pts = P_opt.copy()
                        best_U = U_opt.copy()
                        best_chart = 'south'
        except Exception:
            pass
    
    # Basin Hopping on the best solution
    if best_pts is not None and time.time() < deadline - 25:
        if best_chart == 'north':
            obj_func = _energy_and_grad_stereo
            stereo_func = _stereo_to_sphere
        else:
            obj_func = _energy_and_grad_stereo_south
            stereo_func = _stereo_to_sphere_south
        
        current_U = best_U.copy()
        current_energy = best_energy
        temp = 0.8
        decay = 0.93
        
        for rh in range(80):
            if time.time() > deadline - 12:
                break
            
            rng = np.random.default_rng(rh + 500)
            noise = rng.standard_normal((N, 2)) * 0.4 * temp
            new_U = current_U + noise
            
            try:
                res = minimize(obj_func, new_U, method='L-BFGS-B', jac=True, 
                               options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-8})
                U_opt = res.x
                P_opt = stereo_func(U_opt)
                P_opt = _normalize(P_opt)
                energy = _compute_energy(P_opt)
                
                delta_e = energy - current_energy
                # Fixed acceptance: always accept improvements, accept worsenings with Metropolis probability
                if delta_e < 0 or (temp > 1e-10 and np.random.default_rng(rh + 1000).random() < math.exp(-delta_e / temp)):
                    current_U = U_opt
                    current_energy = energy
                    if energy < best_energy:
                        best_energy = energy
                        best_pts = P_opt.copy()
                        best_U = U_opt.copy()
                        
            except Exception:
                pass
                
            temp *= decay
            
    # Final Polish with tight tolerances
    if best_pts is not None and time.time() < deadline - 5:
        if best_chart == 'north':
            obj_func = _energy_and_grad_stereo
            stereo_func = _stereo_to_sphere
        else:
            obj_func = _energy_and_grad_stereo_south
            stereo_func = _stereo_to_sphere_south
        
        try:
            res = minimize(obj_func, best_U, method='L-BFGS-B', jac=True, 
                           options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50})
            U_opt = res.x
            P_opt = stereo_func(U_opt)
            P_opt = _normalize(P_opt)
            energy = _compute_energy(P_opt)
            
            if energy < best_energy:
                best_energy = energy
                best_pts = P_opt.copy()
                
        except Exception:
            pass
            
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
        
    return _normalize(best_pts)
