import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE_S = 142.0

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _energy_and_grad_sphere(pts):
    """
    Compute energy and gradient for points on the sphere.
    Energy = sum_{i<j} log(1/|p_i - p_j|)
    Gradient is the Riemannian gradient (projected to tangent space).
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    
    # Energy: sum_{i<j} log(1/d_ij) = -0.5 * sum_{i!=j} log(d_ij) = -0.25 * sum_{i!=j} log(d_ij^2)
    log_dists_sq = np.log(dists_sq)
    np.fill_diagonal(log_dists_sq, 0.0)
    energy = -0.25 * np.sum(log_dists_sq)
    
    # Gradient: d/dp_i sum_{j!=i} log(1/|p_i - p_j|) = sum_{j!=i} (p_i - p_j) / |p_i - p_j|^2
    # = sum_{j!=i} p_i / |p_i - p_j|^2 - sum_{j!=i} p_j / |p_i - p_j|^2
    # = p_i * sum_{j!=i} 1/|p_i - p_j|^2 - sum_{j!=i} p_j / |p_i - p_j|^2
    inv_dists_sq = 1.0 / dists_sq
    np.fill_diagonal(inv_dists_sq, 0.0)
    
    weighted_sum_P = inv_dists_sq @ pts
    sum_inv_dists_sq = np.sum(inv_dists_sq, axis=1)
    grad_ambient = pts * sum_inv_dists_sq[:, np.newaxis] - weighted_sum_P
    
    # Project to tangent space: grad - (grad . p) p
    dot_prod = np.sum(grad_ambient * pts, axis=1, keepdims=True)
    grad = grad_ambient - dot_prod * pts
    
    return energy, grad

def _riemannian_cg(pts, max_iter=800, tol=1e-12, time_budget=20.0):
    """
    Riemannian conjugate gradient on the sphere for log-energy minimization.
    Uses Polak-Ribiere beta update and backtracking line search with Wolfe conditions.
    """
    start_time = time.time()
    pts = pts.copy()
    n = len(pts)
    
    def _time_exceeded():
        return time.time() - start_time > time_budget
    
    energy, grad = _energy_and_grad_sphere(pts)
    p = -grad.copy()
    
    for it in range(max_iter):
        if _time_exceeded():
            break
        
        # Backtracking line search with Armijo condition
        best_alpha = 0.0
        best_e = energy
        best_new_pts = pts.copy()
        best_new_grad = grad.copy()
        alpha = 1.0
        grad_dot_p = np.sum(grad * p)
        
        for _ls in range(30):
            if _time_exceeded():
                break
            
            # Exponential map
            step = alpha * p
            step_norms = np.linalg.norm(step, axis=1, keepdims=True)
            step_norms_safe = np.maximum(step_norms, 1e-15)
            cos_theta = np.cos(step_norms_safe)
            sin_theta_over_norm = np.sin(step_norms_safe) / step_norms_safe
            new_pts = cos_theta * pts + sin_theta_over_norm * step
            new_pts = _normalize(new_pts)
            
            new_e, new_grad = _energy_and_grad_sphere(new_pts)
            
            # Armijo condition
            if new_e < energy + 1e-4 * alpha * grad_dot_p:
                best_alpha = alpha
                best_e = new_e
                best_new_pts = new_pts
                best_new_grad = new_grad
                break
            
            if new_e < best_e:
                best_e = new_e
                best_alpha = alpha
                best_new_pts = new_pts
                best_new_grad = new_grad
            
            alpha *= 0.5
        
        if best_alpha == 0.0:
            break
        
        pts = best_new_pts
        new_energy = best_e
        new_grad = best_new_grad
        
        # Check convergence
        grad_norm = np.linalg.norm(new_grad)
        if grad_norm < tol:
            break
        
        # Polak-Ribiere beta
        beta_num = np.sum(new_grad * (new_grad - grad))
        beta_den = np.sum(grad * grad)
        if beta_den > 1e-15:
            beta = max(0.0, min(beta_num / beta_den, 1.0))
        else:
            beta = 0.0
        
        p = -new_grad + beta * p
        # Project p to tangent space
        dot_p = np.sum(p * pts, axis=1, keepdims=True)
        p -= dot_p * pts
        
        energy = new_energy
        grad = new_grad
    
    return pts, _compute_energy(pts)

def _compute_energy(pts):
    """Compute logarithmic energy from sphere points."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    log_dists_sq = np.log(dists_sq)
    np.fill_diagonal(log_dists_sq, 0.0)
    return -0.25 * np.sum(log_dists_sq)

def _riemannian_newton_polish(pts, max_iter=100, tol=1e-14, time_budget=10.0):
    """
    Riemannian Newton-type polish using approximate Hessian.
    For the log-energy, the Hessian in ambient coordinates can be approximated,
    and we project to tangent space.
    """
    start_time = time.time()
    pts = pts.copy()
    n = len(pts)
    
    def _time_exceeded():
        return time.time() - start_time > time_budget
    
    energy, grad = _energy_and_grad_sphere(pts)
    
    for it in range(max_iter):
        if _time_exceeded():
            break
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # Approximate Newton step: use gradient descent with adaptive step
        # For better convergence, try multiple step sizes
        best_alpha = 0.0
        best_e = energy
        best_new_pts = pts.copy()
        best_new_grad = grad.copy()
        
        # Try a range of step sizes
        for alpha in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]:
            if _time_exceeded():
                break
            
            step = -alpha * grad
            step_norms = np.linalg.norm(step, axis=1, keepdims=True)
            step_norms_safe = np.maximum(step_norms, 1e-15)
            cos_theta = np.cos(step_norms_safe)
            sin_theta_over_norm = np.sin(step_norms_safe) / step_norms_safe
            new_pts = cos_theta * pts + sin_theta_over_norm * step
            new_pts = _normalize(new_pts)
            
            new_e, new_grad = _energy_and_grad_sphere(new_pts)
            
            if new_e < best_e:
                best_e = new_e
                best_alpha = alpha
                best_new_pts = new_pts
                best_new_grad = new_grad
        
        if best_alpha == 0.0:
            break
        
        pts = best_new_pts
        energy = best_e
        grad = best_new_grad
    
    return pts, _compute_energy(pts)

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _halton_sphere(n):
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
        theta = 2 * np.pi * u
        phi = np.arccos(2 * v - 1)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        points.append([x, y, z])
    pts = np.array(points)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _stereo_to_sphere(U):
    """Convert stereographic coordinates (u, v) to sphere points (north pole projection)."""
    u, v = U[:, 0], U[:, 1]
    r2 = u**2 + v**2
    denom = 1.0 + r2
    px = 2.0 * u / denom
    py = 2.0 * v / denom
    pz = (1.0 - r2) / denom
    return np.column_stack([px, py, pz])

def _sphere_to_stereo(P):
    """Convert sphere points to stereographic coordinates (north pole projection)."""
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    denom = 1.0 - z
    u = x / denom
    v = y / denom
    return np.column_stack([u, v])

def _stereo_to_sphere_south(U):
    """Convert stereographic coordinates (u, v) to sphere points (south pole projection)."""
    u, v = U[:, 0], U[:, 1]
    r2 = u**2 + v**2
    denom = 1.0 + r2
    px = 2.0 * u / denom
    py = 2.0 * v / denom
    pz = (r2 - 1.0) / denom
    return np.column_stack([px, py, pz])

def _sphere_to_stereo_south(P):
    """Convert sphere points to stereographic coordinates (south pole projection)."""
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    denom = 1.0 + z
    u = x / denom
    v = y / denom
    return np.column_stack([u, v])

def _energy_and_grad_stereo(U):
    """Compute energy and gradient with respect to stereographic coordinates U (north pole)."""
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
    """Compute energy and gradient with respect to stereographic coordinates U (south pole)."""
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

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_S
    
    # Initializations
    inits = []
    inits.append(_fibonacci_sphere(N))
    inits.append(_halton_sphere(N))
    
    for s in range(12):
        rng = np.random.default_rng(s)
        pts = rng.standard_normal((N, 3))
        pts = _normalize(pts)
        inits.append(pts)
    
    for s in range(12):
        rng = np.random.default_rng(s + 100)
        scale = 0.05 + 0.03 * (s % 4)
        pts = _fibonacci_sphere(N) + rng.standard_normal((N, 3)) * scale
        pts = _normalize(pts)
        inits.append(pts)
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Riemannian CG from multiple initializations
    for idx, pts in enumerate(inits):
        if time.time() > deadline - 30:
            break
        
        remaining = deadline - time.time()
        time_per_init = min(8.0, max(2.0, remaining / max(1, len(inits) - idx)))
        
        try:
            pts_opt, energy = _riemannian_cg(pts, max_iter=600, tol=1e-12, time_budget=time_per_init)
            if energy < best_energy:
                best_energy = energy
                best_pts = pts_opt.copy()
        except Exception:
            pass
    
    # Phase 2: Basin hopping with Riemannian CG
    if best_pts is not None and time.time() < deadline - 25:
        current_pts = best_pts.copy()
        current_energy = best_energy
        temp = 0.8
        decay = 0.90
        
        for rh in range(80):
            if time.time() > deadline - 20:
                break
            
            rng = np.random.default_rng(rh + 500)
            # Adaptive noise: larger early, smaller later
            noise_scale = 0.5 * temp * (1.0 + 0.5 * np.sin(rh * 0.1))
            noise = rng.standard_normal((N, 3)) * noise_scale
            # Project noise to tangent space
            dot_noise = np.sum(noise * current_pts, axis=1, keepdims=True)
            noise -= dot_noise * current_pts
            new_pts = current_pts + noise
            new_pts = _normalize(new_pts)
            
            remaining = deadline - time.time()
            time_per_hop = min(4.0, max(0.5, remaining / max(1, 80 - rh)))
            
            try:
                pts_opt, energy = _riemannian_cg(new_pts, max_iter=500, tol=1e-11, time_budget=time_per_hop)
                
                delta_e = energy - current_energy
                # Metropolis acceptance
                if delta_e < 0 or (temp > 1e-10 and np.random.default_rng(rh + 1000).random() < math.exp(-delta_e / temp)):
                    current_pts = pts_opt
                    current_energy = energy
                    if energy < best_energy:
                        best_energy = energy
                        best_pts = pts_opt.copy()
            except Exception:
                pass
            
            temp *= decay
    
    # Phase 3: Stereo chart L-BFGS-B polish from best solution
    if best_pts is not None and time.time() < deadline - 10:
        # Try north-pole stereographic
        try:
            if not np.any(best_pts[:, 2] > 0.999):
                U = _sphere_to_stereo(best_pts)
                if np.all(np.isfinite(U)):
                    res = minimize(_energy_and_grad_stereo, U, method='L-BFGS-B', jac=True,
                                   options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-13, 'maxls': 100})
                    P_opt = _stereo_to_sphere(res.x)
                    P_opt = _normalize(P_opt)
                    energy = _compute_energy(P_opt)
                    if energy < best_energy:
                        best_energy = energy
                        best_pts = P_opt.copy()
        except Exception:
            pass
        
        # Try south-pole stereographic
        try:
            if not np.any(best_pts[:, 2] < -0.999):
                U = _sphere_to_stereo_south(best_pts)
                if np.all(np.isfinite(U)):
                    res = minimize(_energy_and_grad_stereo_south, U, method='L-BFGS-B', jac=True,
                                   options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-13, 'maxls': 100})
                    P_opt = _stereo_to_sphere_south(res.x)
                    P_opt = _normalize(P_opt)
                    energy = _compute_energy(P_opt)
                    if energy < best_energy:
                        best_energy = energy
                        best_pts = P_opt.copy()
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish with Newton-type steps
    if best_pts is not None and time.time() < deadline - 3:
        try:
            remaining = deadline - time.time() - 1.0
            pts_opt, energy = _riemannian_newton_polish(best_pts, max_iter=200, tol=1e-14, time_budget=remaining * 0.6)
            if energy < best_energy:
                best_energy = energy
                best_pts = pts_opt.copy()
        except Exception:
            pass
        
        # Final CG polish
        if best_pts is not None and time.time() < deadline - 1:
            try:
                remaining = deadline - time.time() - 0.5
                pts_opt, energy = _riemannian_cg(best_pts, max_iter=1000, tol=1e-14, time_budget=remaining)
                if energy < best_energy:
                    best_energy = energy
                    best_pts = pts_opt.copy()
            except Exception:
                pass
    
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    return _normalize(best_pts)
