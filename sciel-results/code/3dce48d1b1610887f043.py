import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _energy_and_grad(theta_phi):
    """
    theta_phi: shape (N, 2), theta in [0, pi], phi in [0, 2pi]
    Returns energy and gradient.
    """
    theta = theta_phi[:, 0]
    phi = theta_phi[:, 1]
    
    # Compute coordinates
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    x = sin_t * np.cos(phi)
    y = sin_t * np.sin(phi)
    z = cos_t
    
    pts = np.column_stack((x, y, z))
    
    # Compute pairwise distances squared
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff**2, axis=2)
    
    # Mask for i < j
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d2_pairs = d2[mask]
    
    # Energy: sum log(1 / d) = - sum log(d) = -0.5 sum log(d2)
    d2_safe = np.maximum(d2_pairs, 1e-12)
    energy = -0.5 * np.sum(np.log(d2_safe))
    
    # Gradient using vectorized computation
    ti = theta[:, np.newaxis]
    tj = theta[np.newaxis, :]
    pi = phi[:, np.newaxis]
    pj = phi[np.newaxis, :]
    
    sin_ti = np.sin(ti)
    cos_ti = np.cos(ti)
    sin_tj = np.sin(tj)
    cos_tj = np.cos(tj)
    
    cos_diff = np.cos(pi - pj)
    sin_diff = np.sin(pi - pj)
    
    # d2 matrix
    d2_mat = 2 - 2 * (sin_ti * sin_tj * cos_diff + cos_ti * cos_tj)
    
    # Derivatives
    d_d2_d_theta_i = -2 * (cos_ti * sin_tj * cos_diff - sin_ti * cos_tj)
    d_d2_d_phi_i = 2 * sin_ti * sin_tj * sin_diff
    
    # Mask diagonal
    mask_diag = ~np.eye(N, dtype=bool)
    
    # Safe d2 for division
    d2_safe_mat = np.where(mask_diag, d2_mat, 1.0)
    
    inv_d2 = np.where(mask_diag, 1.0 / d2_safe_mat, 0.0)
    
    grad_theta = -0.5 * np.sum(inv_d2 * d_d2_d_theta_i, axis=1)
    grad_phi = -0.5 * np.sum(inv_d2 * d_d2_d_phi_i, axis=1)
    
    grad = np.column_stack((grad_theta, grad_phi))
    
    return energy, grad

def _reconstruct(theta_phi):
    theta = theta_phi[:, 0]
    phi = theta_phi[:, 1]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack((x, y, z))

def _polish_steepest_descent(theta_phi, max_iter=200, tol=1e-14):
    """
    Final polish using steepest descent with line search in theta-phi space.
    This helps converge to machine precision.
    """
    tp = theta_phi.copy()
    energy, grad = _energy_and_grad(tp)
    
    for it in range(max_iter):
        if not np.all(np.isfinite(grad)):
            break
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # Line search: find step size alpha that minimizes energy
        best_alpha = 0.01
        best_e = energy
        alphas = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
        
        for alpha in alphas:
            tp_trial = tp - alpha * grad
            # Clip theta to valid range
            tp_trial[:, 0] = np.clip(tp_trial[:, 0], 1e-8, np.pi - 1e-8)
            tp_trial[:, 1] = (tp_trial[:, 1] + np.pi) % (2 * np.pi) - np.pi
            tp_trial[:, 1] = np.where(tp_trial[:, 1] < 0, tp_trial[:, 1] + 2 * np.pi, tp_trial[:, 1])
            
            e_trial, _ = _energy_and_grad(tp_trial)
            if e_trial < best_e:
                best_e = e_trial
                best_alpha = alpha
        
        if best_alpha == 0.0:
            break
        
        tp = tp - best_alpha * grad
        tp[:, 0] = np.clip(tp[:, 0], 1e-8, np.pi - 1e-8)
        tp[:, 1] = (tp[:, 1] + np.pi) % (2 * np.pi) - np.pi
        tp[:, 1] = np.where(tp[:, 1] < 0, tp[:, 1] + 2 * np.pi, tp[:, 1])
        
        energy, grad = _energy_and_grad(tp)
    
    return tp, energy

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin before 120s limit
    
    best_energy = np.inf
    best_pts = None
    best_theta_phi = None
    
    # Initializations
    inits = []
    
    # 1. Fibonacci
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    # Convert to theta, phi
    theta_fib = np.arccos(z)
    phi_fib = np.arctan2(y, x)
    # Handle negative phi
    phi_fib = np.where(phi_fib < 0, phi_fib + 2*np.pi, phi_fib)
    inits.append(np.column_stack((theta_fib, phi_fib)))
    
    # 2. Random uniform on sphere with different seeds
    for seed in [42, 123, 456, 789, 1000, 2024]:
        np.random.seed(seed)
        v = np.random.randn(N, 3)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        theta_rand = np.arccos(v[:, 2])
        phi_rand = np.arctan2(v[:, 1], v[:, 0])
        phi_rand = np.where(phi_rand < 0, phi_rand + 2*np.pi, phi_rand)
        inits.append(np.column_stack((theta_rand, phi_rand)))
        
    # 3. Perturbed Fibonacci with different scales
    for scale in [0.05, 0.1, 0.2, 0.5]:
        np.random.seed(int(scale * 1000))
        theta_p = theta_fib + np.random.randn(N) * scale
        phi_p = phi_fib + np.random.randn(N) * scale
        theta_p = np.clip(theta_p, 1e-6, np.pi - 1e-6)
        phi_p = np.where(phi_p < 0, phi_p + 2*np.pi, phi_p)
        inits.append(np.column_stack((theta_p, phi_p)))
    
    # 4. Icosahedron-based: start from 12 icosahedron vertices and subdivide
    # Simple approach: use golden ratio to create well-distributed points
    np.random.seed(999)
    for k in range(3):
        v = np.random.randn(N, 3)
        # Apply a random rotation to break symmetry
        R = np.linalg.qr(np.random.randn(3, 3))[0]
        v = v @ R.T
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        theta_rot = np.arccos(v[:, 2])
        phi_rot = np.arctan2(v[:, 1], v[:, 0])
        phi_rot = np.where(phi_rot < 0, phi_rot + 2*np.pi, phi_rot)
        inits.append(np.column_stack((theta_rot, phi_rot)))
        
    # Optimize each initialization
    for idx, init in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
            
        try:
            # Phase 1: L-BFGS-B with moderate tolerances
            res = minimize(
                _energy_and_grad, 
                init, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-7}
            )
            energy = res.fun
            
            if energy < best_energy:
                best_energy = energy
                best_theta_phi = res.x.copy()
                best_pts = _reconstruct(res.x)
        except Exception:
            continue
        
        # Phase 2: Polish with tighter tolerances
        if time.time() > deadline - 15.0:
            break
            
        try:
            polished_tp, polished_e = _polish_steepest_descent(res.x, max_iter=100, tol=1e-13)
            if polished_e < best_energy:
                best_energy = polished_e
                best_theta_phi = polished_tp.copy()
                best_pts = _reconstruct(polished_tp)
        except Exception:
            pass
        
        # Phase 3: Another L-BFGS-B polish with very tight tolerances
        if time.time() > deadline - 10.0:
            break
            
        try:
            res2 = minimize(
                _energy_and_grad, 
                best_theta_phi if best_theta_phi is not None else res.x, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            energy2 = res2.fun
            if energy2 < best_energy:
                best_energy = energy2
                best_theta_phi = res2.x.copy()
                best_pts = _reconstruct(res2.x)
        except Exception:
            pass
        
        # Phase 4: Final steepest descent polish
        if time.time() > deadline - 5.0:
            break
            
        try:
            final_tp, final_e = _polish_steepest_descent(best_theta_phi, max_iter=50, tol=1e-14)
            if final_e < best_energy:
                best_energy = final_e
                best_pts = _reconstruct(final_tp)
        except Exception:
            pass
            
    # Fallback to Fibonacci if no optimization succeeded
    if best_pts is None:
        best_pts = _reconstruct(inits[0])
        
    # Ensure all points are on the unit sphere and valid
    norms = np.linalg.norm(best_pts, axis=1)
    best_pts = best_pts / norms[:, np.newaxis]
    
    # Check for very close points and separate them if needed
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Perturb point j slightly
                best_pts[j] = best_pts[j] + np.random.randn(3) * 1e-8
                best_pts[j] /= np.linalg.norm(best_pts[j])
    
    return best_pts
