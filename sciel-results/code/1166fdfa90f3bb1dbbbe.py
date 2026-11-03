import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin

    # 1. Define the energy and gradient functions in spherical coordinates
    # theta: (N,) in [0, pi]
    # phi: (N,) in [0, 2*pi)
    
    def compute_points(theta, phi):
        st = np.sin(theta)
        ct = np.cos(theta)
        sp = np.sin(phi)
        cp = np.cos(phi)
        x = st * cp
        y = st * sp
        z = ct
        return np.column_stack([x, y, z])

    def energy_and_grad(params):
        theta = params[:N]
        phi = params[N:]
        
        st = np.sin(theta)
        ct = np.cos(theta)
        sp = np.sin(phi)
        cp = np.cos(phi)
        x = st * cp
        y = st * sp
        z = ct
        pts = np.column_stack([x, y, z])
        
        dots = pts @ pts.T
        d2 = 2.0 - 2.0 * dots
        np.fill_diagonal(d2, 1.0)
        
        iu = np.triu_indices(N, 1)
        d2_u = d2[iu]
        
        inv_d2 = 1.0 / d2
        np.fill_diagonal(inv_d2, 0.0)
        
        sum_x_inv_d2 = pts.T @ inv_d2
        sum_inv_d2 = np.sum(inv_d2, axis=1)
        
        x_i_term = pts * sum_inv_d2[:, np.newaxis]
        grad_x = sum_x_inv_d2.T - x_i_term
        
        dx_dtheta = np.column_stack([ct * cp, ct * sp, -st])
        grad_theta = np.sum(grad_x * dx_dtheta, axis=1)
        
        dx_dphi = np.column_stack([-st * sp, st * cp, np.zeros(N)])
        grad_phi = np.sum(grad_x * dx_dphi, axis=1)
        
        grad = np.concatenate([grad_theta, grad_phi])
        
        energy = -0.5 * np.sum(np.log(d2_u))
        
        return energy, grad

    def compute_cartesian_grad(pts):
        dots = pts @ pts.T
        d2 = 2.0 - 2.0 * dots
        np.fill_diagonal(d2, 1.0)
        inv_d2 = 1.0 / d2
        np.fill_diagonal(inv_d2, 0.0)
        
        sum_x_inv_d2 = pts.T @ inv_d2
        sum_inv_d2 = np.sum(inv_d2, axis=1)
        
        x_i_term = pts * sum_inv_d2[:, np.newaxis]
        grad_x = sum_x_inv_d2.T - x_i_term
        return grad_x

    def riemannian_step(pts, grad_x, step_size):
        x_new = pts - step_size * grad_x
        norms = np.linalg.norm(x_new, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        x_new = x_new / norms
        return x_new

    # 2. Initialization strategies
    candidates = []
    
    # Strategy A: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    params_init_A = np.concatenate([theta, phi])
    candidates.append(params_init_A)
    
    # Strategy B: Jittered Fibonacci (reduced count to allow time for GD)
    for _ in range(3):
        if time.time() > deadline - 40: break
        theta_jit = theta + np.random.uniform(-0.1, 0.1, N)
        phi_jit = phi + np.random.uniform(-0.1, 0.1, N)
        theta_jit = np.clip(theta_jit, 1e-6, np.pi - 1e-6)
        phi_jit = np.mod(phi_jit, 2 * np.pi)
        candidates.append(np.concatenate([theta_jit, phi_jit]))
        
    # Strategy C: Random
    for _ in range(3):
        if time.time() > deadline - 40: break
        theta_r = np.random.uniform(0, np.pi, N)
        phi_r = np.random.uniform(0, 2 * np.pi, N)
        candidates.append(np.concatenate([theta_r, phi_r]))

    best_energy = float('inf')
    best_params = None
    best_points = None

    # 3. Optimization loop with Cartesian GD pre-step
    for p_init in candidates:
        if time.time() > deadline - 20:
            break
            
        try:
            # Pre-optimization: Riemannian Gradient Descent in Cartesian space
            theta_init = p_init[:N]
            phi_init = p_init[N:]
            pts_init = compute_points(theta_init, phi_init)
            
            # Perform a few GD steps to move away from bad local minima
            # Start with a moderate step size and decay it
            step_size = 0.1
            for step in range(50):
                if time.time() > deadline - 25: break
                grad_x = compute_cartesian_grad(pts_init)
                pts_init = riemannian_step(pts_init, grad_x, step_size)
                step_size *= 0.95 # Decay step size
                
            # Convert back to spherical for L-BFGS
            # theta = arccos(z), phi = arctan2(y, x)
            z_c = pts_init[:, 2]
            x_c = pts_init[:, 0]
            y_c = pts_init[:, 1]
            theta_opt = np.arccos(np.clip(z_c, -1.0, 1.0))
            phi_opt = np.arctan2(y_c, x_c)
            phi_opt = np.mod(phi_opt, 2 * np.pi)
            
            params_opt = np.concatenate([theta_opt, phi_opt])
            
            res = minimize(
                energy_and_grad,
                params_opt,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if res.success or res.fun < best_energy:
                theta_final = np.clip(res.x[:N], 1e-6, np.pi - 1e-6)
                phi_final = np.mod(res.x[N:], 2 * np.pi)
                params_final = np.concatenate([theta_final, phi_final])
                
                e_val, _ = energy_and_grad(params_final)
                
                if e_val < best_energy:
                    best_energy = e_val
                    best_params = params_final
                    best_points = compute_points(theta_final, phi_final)
                    
        except Exception:
            continue

    # 4. Final Polish
    if best_params is not None and time.time() < deadline - 5:
        try:
            res_final = minimize(
                energy_and_grad,
                best_params,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 50}
            )
            theta_final = np.clip(res_final.x[:N], 1e-6, np.pi - 1e-6)
            phi_final = np.mod(res_final.x[N:], 2 * np.pi)
            params_final = np.concatenate([theta_final, phi_final])
            e_final, _ = energy_and_grad(params_final)
            
            if e_final < best_energy:
                best_energy = e_final
                best_points = compute_points(theta_final, phi_final)
        except Exception:
            pass

    if best_points is None:
        best_points = np.column_stack([x, y, z])

    norms = np.linalg.norm(best_points, axis=1)
    best_points = best_points / norms[:, np.newaxis]

    return best_points
