import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # 1. Define the energy and gradient functions in spherical coordinates
    # theta: (N,) in [0, pi]
    # phi: (N,) in [0, 2*pi)
    
    def compute_points(theta, phi):
        # theta, phi are 1D arrays of length N
        st = np.sin(theta)
        ct = np.cos(theta)
        sp = np.sin(phi)
        cp = np.cos(phi)
        x = st * cp
        y = st * sp
        z = ct
        return np.column_stack([x, y, z])

    def energy_and_grad(params):
        # params is a 1D array of length 2*N: [theta_0..theta_N-1, phi_0..phi_N-1]
        theta = params[:N]
        phi = params[N:]
        
        # Compute points
        st = np.sin(theta)
        ct = np.cos(theta)
        sp = np.sin(phi)
        cp = np.cos(phi)
        x = st * cp
        y = st * sp
        z = ct
        pts = np.column_stack([x, y, z])
        
        # Compute pairwise distances and energy
        # d_ij^2 = 2 - 2 * dot(x_i, x_j)
        dots = pts @ pts.T
        d2 = 2.0 - 2.0 * dots
        # Avoid division by zero if points coincide
        np.fill_diagonal(d2, 1.0) # dummy value for diagonal, we won't use it
        
        # Energy: sum_{i<j} log(1/sqrt(d2_ij)) = -0.5 * sum_{i<j} log(d2_ij)
        # We only need upper triangle
        iu = np.triu_indices(N, 1)
        d2_u = d2[iu]
        
        # For gradient, we need d2 for all pairs (i,j) with i!=j
        # Actually, the formula for grad w.r.t x_i is sum_{j!=i} (x_j - x_i) / d2_ij
        # Let's compute the gradient w.r.t x first, then project to theta, phi
        
        # grad_x[i] = sum_{j != i} (x_j - x_i) / d2_ij
        # Note: (x_j - x_i) / d2_ij = (x_j - x_i) / (2 - 2 x_i.x_j)
        
        # Vectorized computation of grad_x
        # num[i,j] = x_j - x_i
        # den[i,j] = d2_ij
        # We want sum_j (x_j - x_i) / d2_ij
        
        # Let's compute sum_j x_j / d2_ij - x_i * sum_j 1/d2_ij
        inv_d2 = 1.0 / d2
        # Set diagonal to 0 to avoid self-interaction
        np.fill_diagonal(inv_d2, 0.0)
        
        # sum_j x_j / d2_ij
        sum_x_inv_d2 = pts.T @ inv_d2  # shape (3, N), result[i] is sum_j x_j[i] / d2_ij
        # sum_j 1/d2_ij
        sum_inv_d2 = np.sum(inv_d2, axis=1) # shape (N,)
        
        # grad_x[i] = sum_x_inv_d2[:, i] - x_i * sum_inv_d2[i]
        # pts is (N, 3), so x_i is pts[i]
        x_i_term = pts * sum_inv_d2[:, np.newaxis]
        grad_x = sum_x_inv_d2.T - x_i_term # shape (N, 3)
        
        # Now project grad_x to theta, phi
        # dE/dtheta_i = grad_x[i] . d(x_i)/dtheta_i
        # d(x_i)/dtheta_i = (ct_i * cp_i, ct_i * sp_i, -st_i)
        dx_dtheta = np.column_stack([ct * cp, ct * sp, -st])
        grad_theta = np.sum(grad_x * dx_dtheta, axis=1)
        
        # dE/dphi_i = grad_x[i] . d(x_i)/dphi_i
        # d(x_i)/dphi_i = (-st_i * sp_i, st_i * cp_i, 0)
        dx_dphi = np.column_stack([-st * sp, st * cp, np.zeros(N)])
        grad_phi = np.sum(grad_x * dx_dphi, axis=1)
        
        # Combine gradient
        grad = np.concatenate([grad_theta, grad_phi])
        
        # Compute energy value
        # E = -0.5 * sum_{i<j} log(d2_ij)
        # Use fsum for higher precision if needed, but np.sum is faster for optimization
        # The evaluator uses math.fsum. Let's use it for the final check, but np.sum for optimization steps.
        energy = -0.5 * np.sum(np.log(d2_u))
        
        return energy, grad

    # 2. Initialization strategies
    candidates = []
    
    # Strategy A: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    
    # Convert to theta, phi
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    # phi is already computed, but need to handle the mapping from (x,y) to phi
    # phi_calc = np.arctan2(y, x)
    # The spiral phi is good.
    params_init_A = np.concatenate([theta, phi])
    candidates.append(params_init_A)
    
    # Strategy B: Jittered Fibonacci
    for _ in range(5):
        if time.time() > deadline - 30: break
        theta_jit = theta + np.random.uniform(-0.1, 0.1, N)
        phi_jit = phi + np.random.uniform(-0.1, 0.1, N)
        theta_jit = np.clip(theta_jit, 1e-6, np.pi - 1e-6)
        phi_jit = np.mod(phi_jit, 2 * np.pi)
        candidates.append(np.concatenate([theta_jit, phi_jit]))
        
    # Strategy C: Random
    for _ in range(5):
        if time.time() > deadline - 30: break
        theta_r = np.random.uniform(0, np.pi, N)
        phi_r = np.random.uniform(0, 2 * np.pi, N)
        candidates.append(np.concatenate([theta_r, phi_r]))

    best_energy = float('inf')
    best_params = None
    best_points = None

    # 3. Optimization loop
    for p_init in candidates:
        if time.time() > deadline - 20:
            break
            
        try:
            res = minimize(
                energy_and_grad,
                p_init,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if res.success or res.fun < best_energy:
                # Verify and refine if needed
                # The optimizer might return slightly out of bounds, clip them
                theta_opt = np.clip(res.x[:N], 1e-6, np.pi - 1e-6)
                phi_opt = np.mod(res.x[N:], 2 * np.pi)
                params_opt = np.concatenate([theta_opt, phi_opt])
                
                # Get final energy with high precision
                e_val, _ = energy_and_grad(params_opt)
                
                # The evaluator uses math.fsum on log(1/d). 
                # My energy function uses -0.5 * sum(log(d2)) = sum(log(1/sqrt(d2))) = sum(log(1/d)).
                # So e_val is the correct value.
                
                if e_val < best_energy:
                    best_energy = e_val
                    best_params = params_opt
                    best_points = compute_points(theta_opt, phi_opt)
                    
        except Exception:
            continue

    # 4. Final Polish
    # If we have time, do a few more iterations with tighter tolerance on the best solution
    if best_params is not None and time.time() < deadline - 5:
        try:
            res_final = minimize(
                energy_and_grad,
                best_params,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 50}
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
        # Fallback to Fibonacci
        best_points = np.column_stack([x, y, z])

    # Normalize just in case
    norms = np.linalg.norm(best_points, axis=1)
    best_points = best_points / norms[:, np.newaxis]

    return best_points
