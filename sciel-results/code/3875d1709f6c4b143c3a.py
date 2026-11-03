import numpy as np
from scipy.special import lpmv
import math

N = 100
L = 15

def construct():
    # 1. Precompute Spherical Harmonics basis and gradients
    # We use real spherical harmonics to keep coefficients real.
    # Y_lm^real(theta, phi)
    # For m > 0: sqrt(2) * Re(Y_lm)
    # For m = 0: Y_l0
    # For m < 0: sqrt(2) * Im(Y_{l|m|})
    
    # Actually, using complex is easier for derivatives if we handle them correctly.
    # Let's use the complex form and define F1 = Re(sum c_lm Y_lm), F2 = Im(sum c_lm Y_lm)
    # c_lm are complex parameters. Total params: 2 * (L+1)^2.
    
    # Indices for l, m
    lm_indices = []
    for l in range(L + 1):
        for m in range(-l, l + 1):
            lm_indices.append((l, m))
    
    N_params = 2 * len(lm_indices)
    
    # Function to evaluate Y_lm and its derivatives w.r.t theta, phi
    # Y_lm(theta, phi) = P_lm(cos(theta)) * exp(i*m*phi) * normalization
    # Normalization: sqrt((2l+1)/(4pi) * (l-m)!/(l+m)!)
    
    def sph_harm_vals(theta, phi):
        """
        Returns:
        Y: array of shape (N_pts, N_lm)
        dY_dtheta: array of shape (N_pts, N_lm)
        dY_dphi: array of shape (N_pts, N_lm)
        """
        N_pts = len(theta)
        Y = np.zeros((N_pts, len(lm_indices)), dtype=complex)
        dY_dtheta = np.zeros((N_pts, len(lm_indices)), dtype=complex)
        dY_dphi = np.zeros((N_pts, len(lm_indices)), dtype=complex)
        
        ct = np.cos(theta)
        st = np.sin(theta)
        exp_i_mp = np.exp(1j * phi) # This is wrong, m varies
        
        for idx, (l, m) in enumerate(lm_indices):
            # Normalization
            norm = np.sqrt((2*l + 1) / (4*np.pi) * np.math.factorial(l-m) / np.math.factorial(l+m))
            # Associated Legendre P_lm(ct)
            # scipy.special.lpmv(m, l, ct)
            P = lpmv(abs(m), l, ct)
            if m < 0:
                P = (-1)**m * np.math.factorial(l+m) / np.math.factorial(l-m) * P
            
            Y_val = norm * P * np.exp(1j * m * phi)
            Y[:, idx] = Y_val
            
            # d/dphi Y = i*m*Y
            dY_dphi[:, idx] = 1j * m * Y_val
            
            # d/dtheta Y = norm * dP/dtheta * exp(i*m*phi)
            # dP_lm(cos(theta))/dtheta = -sin(theta) * dP_lm(x)/dx
            # dP_lm(x)/dx = (l+1) * P_{l+1,m}(x) - (l+1+m) * x * P_{l,m}(x) / (x^2-1) ?
            # Simpler: use recurrence or numerical?
            # Let's use the derivative formula:
            # d/dtheta P_lm(cos(theta)) = -sin(theta) * [ (l+1) P_{l+1,m}(cos(theta)) - (l+1+m) cos(theta) P_{l,m}(cos(theta)) ] / (cos^2(theta)-1) ? No.
            # Standard derivative:
            # dP_lm(x)/dx = (l+1) P_{l+1,m}(x) - (l+1+m) x P_{l,m}(x) / (x^2-1) is for specific cases?
            # Actually: d/dx P_l^m(x) = 0.5 * (P_{l-1}^{m+1}(x) - P_{l+1}^{m+1}(x))
            # Let's use this.
            
            # dP/dtheta = -sin(theta) * dP/dx
            # dP/dx = 0.5 * (P_{l-1}^{m+1}(x) - P_{l+1}^{m+1}(x))
            
            if l > 0:
                P_l_minus_1_m_plus_1 = lpmv(abs(m+1), l-1, ct)
                if m + 1 < 0:
                    P_l_minus_1_m_plus_1 = (-1)**(m+1) * np.math.factorial(l-1+m+1) / np.math.factorial(l-1-(m+1)) * P_l_minus_1_m_plus_1
            else:
                P_l_minus_1_m_plus_1 = 0.0 # l=0, m=0, derivative is 0
            
            if l < L: # We need l+1, but lpmv works for any l
                P_l_plus_1_m_plus_1 = lpmv(abs(m+1), l+1, ct)
                if m + 1 < 0:
                    P_l_plus_1_m_plus_1 = (-1)**(m+1) * np.math.factorial(l+1+m+1) / np.math.factorial(l+1-(m+1)) * P_l_plus_1_m_plus_1
            else:
                P_l_plus_1_m_plus_1 = 0.0 # This shouldn't happen if we precompute up to L+1? 
                # Wait, lpmv is defined for any l.
            
            dP_dx = 0.5 * (P_l_minus_1_m_plus_1 - P_l_plus_1_m_plus_1)
            dP_dtheta = -st * dP_dx
            
            dY_dtheta[:, idx] = norm * dP_dtheta * np.exp(1j * m * phi)
            
        return Y, dY_dtheta, dY_dphi

    # 2. Define F(x; c)
    # c is real vector of size N_params.
    # c[0::2] = Re(c_lm), c[1::2] = Im(c_lm)
    
    def eval_F_and_J(theta, phi, c):
        """
        Evaluates F1, F2 and Jacobian J (2x2 per point)
        F1 = Re( sum c_lm Y_lm )
        F2 = Im( sum c_lm Y_lm )
        """
        N_pts = len(theta)
        Y, dY_dtheta, dY_dphi = sph_harm_vals(theta, phi)
        
        # c_lm complex
        c_lm = c[0::2] + 1j * c[1::2]
        
        # Sum c_lm * Y_lm
        S = np.sum(c_lm[None, :] * Y, axis=1) # Shape (N_pts,)
        
        F1 = np.real(S)
        F2 = np.imag(S)
        
        # dF1/dtheta = Re( sum c_lm dY_dtheta )
        # dF1/dphi = Re( sum c_lm dY_dphi )
        # dF2/dtheta = Im( sum c_lm dY_dtheta )
        # dF2/dphi = Im( sum c_lm dY_dphi )
        
        dS_dtheta = np.sum(c_lm[None, :] * dY_dtheta, axis=1)
        dS_dphi = np.sum(c_lm[None, :] * dY_dphi, axis=1)
        
        J = np.zeros((N_pts, 2, 2))
        J[:, 0, 0] = np.real(dS_dtheta)
        J[:, 0, 1] = np.real(dS_dphi)
        J[:, 1, 0] = np.imag(dS_dtheta)
        J[:, 1, 1] = np.imag(dS_dphi)
        
        return F1, F2, J

    # 3. Initialization
    np.random.seed(42)
    c = np.random.randn(N_params) * 0.1
    
    # Find initial zeros
    # Grid
    theta_grid = np.linspace(0.1, np.pi - 0.1, 200)
    phi_grid = np.linspace(0, 2*np.pi, 200)
    
    theta_init, phi_init = np.meshgrid(theta_grid, phi_grid)
    theta_flat = theta_init.flatten()
    phi_flat = phi_init.flatten()
    
    F1, F2, J = eval_F_and_J(theta_flat, phi_flat, c)
    dist = np.sqrt(F1**2 + F2**2)
    
    # Find local minima of dist
    # Simple approach: pick points where dist is small
    threshold = np.percentile(dist, 5) # Top 5% smallest
    mask = dist < threshold
    
    if np.sum(mask) < N:
        # If not enough, take all
        mask = np.ones(len(dist), dtype=bool)
        
    # Get indices
    idxs = np.where(mask)[0]
    
    # Refine with Newton
    theta_pts = theta_flat[idxs].copy()
    phi_pts = phi_flat[idxs].copy()
    
    # Keep only N points (random selection if too many)
    if len(theta_pts) > N:
        choice = np.random.choice(len(theta_pts), N, replace=False)
        theta_pts = theta_pts[choice]
        phi_pts = phi_pts[choice]
        
    # If we have fewer than N, add random points?
    # For now, assume we have enough.
    
    # 4. Optimization Loop
    # Adam Optimizer
    m_c = np.zeros_like(c)
    v_c = np.zeros_like(c)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    lr = 0.01
    t = 0
    
    for step in range(100):
        # Evaluate F and J at current points
        F1, F2, J = eval_F_and_J(theta_pts, phi_pts, c)
        
        # Newton Step to update points (zeros of current F)
        # Solve J * delta = -F
        # J is (N, 2, 2), F is (N, 2)
        
        # Inverse of 2x2
        det_J = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        # Avoid division by zero
        det_J[abs(det_J) < 1e-10] = 1e-10
        
        inv_J = np.zeros_like(J)
        inv_J[:, 0, 0] = J[:, 1, 1] / det_J
        inv_J[:, 0, 1] = -J[:, 0, 1] / det_J
        inv_J[:, 1, 0] = -J[:, 1, 0] / det_J
        inv_J[:, 1, 1] = J[:, 0, 0] / det_J
        
        # delta = - inv_J @ F
        F_vec = np.column_stack((F1, F2))
        delta = np.sum(inv_J * (-F_vec)[:, :, None], axis=1) # (N, 2)
        
        theta_pts -= delta[:, 0]
        phi_pts -= delta[:, 1]
        
        # Wrap phi
        phi_pts = phi_pts % (2 * np.pi)
        
        # Calculate Energy and Gradient
        # Convert to Cartesian
        st = np.sin(theta_pts)
        x = st * np.cos(phi_pts)
        y = st * np.sin(phi_pts)
        z = np.cos(theta_pts)
        pts = np.column_stack((x, y, z))
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        d = dists[iu]
        
        if np.any(d < 1e-8):
            # Penalize
            E = 1e10
            grad_c = np.zeros_like(c)
        else:
            E = -np.sum(np.log(d))
            
            # Gradient of E w.r.t x_i
            # dE/dx_i = sum_{j != i} (x_i - x_j) / d_ij^2
            grad_x = np.zeros_like(pts)
            for i in range(N):
                for j in range(N):
                    if i != j:
                        grad_x[i] += (pts[i] - pts[j]) / (dists[i, j]**2)
                        
            # Gradient of E w.r.t theta, phi
            # x = st ct, y = st sp, z = ct
            # dx/dtheta = ct ct, dx/dphi = -st sp
            # dy/dtheta = ct sp, dy/dphi = st cp
            # dz/dtheta = -st, dz/dphi = 0
            
            ct = np.cos(theta_pts)
            sp = np.sin(phi_pts)
            cp = np.cos(phi_pts)
            
            dtheta = np.zeros(N)
            dphi = np.zeros(N)
            
            for i in range(N):
                dtheta[i] = grad_x[i, 0] * ct[i]*ct[i] + grad_x[i, 1] * ct[i]*sp[i] + grad_x[i, 2] * (-st[i])
                dphi[i] = grad_x[i, 0] * (-st[i]*sp[i]) + grad_x[i, 1] * (st[i]*cp[i])
                
            grad_theta_phi = np.column_stack((dtheta, dphi)) # (N, 2)
            
            # Gradient of F w.r.t c
            # dF/dc = Y (complex)
            # We need dF1/dc and dF2/dc
            # F1 = Re(S), F2 = Im(S)
            # S = sum c_lm Y_lm
            # dS/dc_re = Y
            # dS/dc_im = i*Y
            # dF1/dc_re = Re(Y)
            # dF1/dc_im = Re(iY) = -Im(Y)
            # dF2/dc_re = Im(Y)
            # dF2/dc_im = Im(iY) = Re(Y)
            
            Y, _, _ = sph_harm_vals(theta_pts, phi_pts)
            
            dF1_dc_re = np.real(Y)
            dF1_dc_im = -np.imag(Y)
            dF2_dc_re = np.imag(Y)
            dF2_dc_im = np.real(Y)
            
            # Chain Rule
            # dE/dc = sum_i (grad_theta_phi_i dot (-J_i^{-1} dF/dc_i))
            
            # We have inv_J (N, 2, 2)
            # delta_theta_phi_dc = - inv_J @ dF/dc
            # dF/dc is (N, 2, N_params) ?
            
            # dF1/dc_k and dF2/dc_k
            # Let's construct dF_dc (N, 2, N_params)
            dF_dc = np.zeros((N, 2, N_params))
            
            # c[0::2] is re, c[1::2] is im
            for k in range(N_params):
                if k % 2 == 0:
                    # re
                    dF_dc[:, 0, k] = dF1_dc_re[:, k//2]
                    dF_dc[:, 1, k] = dF2_dc_re[:, k//2]
                else:
                    # im
                    dF_dc[:, 0, k] = dF1_dc_im[:, k//2]
                    dF_dc[:, 1, k] = dF2_dc_im[:, k//2]
                    
            # delta = - inv_J @ dF_dc
            # inv_J is (N, 2, 2), dF_dc is (N, 2, P)
            # Result (N, P)
            delta_theta_phi_dc = np.sum(inv_J[:, :, None] * (-dF_dc), axis=1)
            
            # grad_c = sum_i grad_theta_phi_i * delta_theta_phi_dc_i
            grad_c = np.sum(grad_theta_phi[:, :, None] * delta_theta_phi_dc, axis=0)
            
        # Update c with Adam
        t += 1
        m_c = beta1 * m_c + (1 - beta1) * grad_c
        v_c = beta2 * v_c + (1 - beta2) * (grad_c ** 2)
        m_hat = m_c / (1 - beta1 ** t)
        v_hat = v_c / (1 - beta2 ** t)
        c -= lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # Normalize c to prevent blowup?
        # c = c / np.linalg.norm(c) * 0.1
        
    # Final Refinement
    F1, F2, J = eval_F_and_J(theta_pts, phi_pts, c)
    det_J = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    det_J[abs(det_J) < 1e-10] = 1e-10
    inv_J = np.zeros_like(J)
    inv_J[:, 0, 0] = J[:, 1, 1] / det_J
    inv_J[:, 0, 1] = -J[:, 0, 1] / det_J
    inv_J[:, 1, 0] = -J[:, 1, 0] / det_J
    inv_J[:, 1, 1] = J[:, 0, 0] / det_J
    F_vec = np.column_stack((F1, F2))
    delta = np.sum(inv_J * (-F_vec)[:, :, None], axis=1)
    theta_pts -= delta[:, 0]
    phi_pts -= delta[:, 1]
    phi_pts = phi_pts % (2 * np.pi)
    
    # Convert to Cartesian
    st = np.sin(theta_pts)
    x = st * np.cos(phi_pts)
    y = st * np.sin(phi_pts)
    z = np.cos(theta_pts)
    pts = np.column_stack((x, y, z))
    
    # Normalize
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    return pts

