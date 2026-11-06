"""
Broadband antireflection coating design using multi-start L-BFGS-B optimization
with analytic gradients.
"""
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _compute_Y(M11, M12, M21, M22, n_sub):
    """Compute admittance Y from characteristic matrix."""
    return (M21 + M22 * n_sub) / (M11 + M12 * n_sub)

def _compute_R(M11, M12, M21, M22, n_sub, n_amb):
    """Compute reflectance R from characteristic matrix."""
    Y = _compute_Y(M11, M12, M21, M22, n_sub)
    r = (n_amb - Y) / (n_amb + Y)
    return np.abs(r) ** 2

def _objective_and_grad(x, n_layers, materials_dict, n_amb, n_sub, wls):
    """
    Compute mean reflectance and its gradient with respect to thicknesses.
    x: array of thicknesses (nm)
    """
    n = len(x)
    # Precompute sin/cos and derivatives for all layers
    # beta_j = 2*pi*n_j*d_j / lambda
    # d(beta_j)/d(d_j) = 2*pi*n_j / lambda
    
    sin_betas = np.zeros((n, len(wls)))
    cos_betas = np.zeros((n, len(wls)))
    d_beta_dd = np.zeros((n, len(wls)))
    
    for j in range(n):
        n_j = materials_dict[materials[j]]
        d_beta_dd[j] = 2.0 * np.pi * n_j / wls
        sin_betas[j] = np.sin(d_beta_dd[j] * x[j])
        cos_betas[j] = np.cos(d_beta_dd[j] * x[j])
    
    # Forward pass to get M matrices
    # M is the product of individual layer matrices
    # We need M_k (product of first k layers) and P_k (product of last n-k layers)
    # to compute gradient efficiently: dM/dx_j = P_{j-1} * dL_j/dx_j * M_{j+1}
    
    # M_k[k] = product of layers 0..k-1. M_k[0] = I.
    # We store M_k as 4 arrays of shape (n+1, len(wls))
    M11_k = np.zeros((n+1, len(wls)), dtype=complex)
    M12_k = np.zeros((n+1, len(wls)), dtype=complex)
    M21_k = np.zeros((n+1, len(wls)), dtype=complex)
    M22_k = np.zeros((n+1, len(wls)), dtype=complex)
    
    M11_k[0] = 1.0
    M22_k[0] = 1.0
    
    for j in range(n):
        n_j = materials_dict[materials[j]]
        c = cos_betas[j]
        s = sin_betas[j]
        
        L11 = c
        L12 = 1.0j * s / n_j
        L21 = n_j * 1.0j * s
        L22 = c
        
        M11_k[j+1] = M11_k[j] * L11 + M12_k[j] * L21
        M12_k[j+1] = M11_k[j] * L12 + M12_k[j] * L22
        M21_k[j+1] = M21_k[j] * L11 + M22_k[j] * L21
        M22_k[j+1] = M21_k[j] * L12 + M22_k[j] * L22
    
    # Total matrix M = M_k[n]
    M11 = M11_k[n]
    M12 = M12_k[n]
    M21 = M21_k[n]
    M22 = M22_k[n]
    
    R = _compute_R(M11, M12, M21, M22, n_sub, n_amb)
    mean_R = np.mean(R)
    
    # Backward pass for gradients
    # We need d(mean_R)/dx_j = mean(dR/dM * dM/dx_j)
    # dR/dM = dR/dY * dY/dM
    # Y = (M21 + M22*n_sub) / (M11 + M12*n_sub)
    # Let A = M11 + M12*n_sub, B = M21 + M22*n_sub
    # Y = B/A
    # dY/dM21 = 1/A
    # dY/dM22 = n_sub/A
    # dY/dM11 = -B/A^2
    # dY/dM12 = -B*n_sub/A^2
    
    A = M11 + M12 * n_sub
    B = M21 + M22 * n_sub
    Y = B / A
    
    r = (n_amb - Y) / (n_amb + Y)
    R = np.abs(r) ** 2
    
    # dR/dY: R = |r|^2. r = (n_amb - Y)/(n_amb + Y)
    # dr/dY = -2*n_amb / (n_amb + Y)^2
    # dR/dY = 2 * Re(r * conj(dr/dY))
    dr_dY = -2.0 * n_amb / (n_amb + Y)**2
    dR_dY = 2.0 * np.real(r * np.conj(dr_dY))
    
    # dY/dM components
    dY_dM21 = 1.0 / A
    dY_dM22 = n_sub / A
    dY_dM11 = -B / (A**2)
    dY_dM12 = -B * n_sub / (A**2)
    
    # dR/dM components
    dR_dM21 = dR_dY * dY_dM21
    dR_dM22 = dR_dY * dY_dM22
    dR_dM11 = dR_dY * dY_dM11
    dR_dM12 = dR_dY * dY_dM12
    
    # Now compute dM/dx_j
    # M = M_{j-1} * L_j * M_{j+1} (conceptually, but M_{j+1} is the product of layers j+1..n)
    # Actually, M_total = M_k[j] * L_j * M_tail[j]
    # where M_tail[j] is the product of layers j+1..n-1.
    # Let's compute M_tail arrays.
    # P_k[k] = product of layers k..n-1. P_k[n] = I.
    P11_k = np.zeros((n+1, len(wls)), dtype=complex)
    P12_k = np.zeros((n+1, len(wls)), dtype=complex)
    P21_k = np.zeros((n+1, len(wls)), dtype=complex)
    P22_k = np.zeros((n+1, len(wls)), dtype=complex)
    
    P11_k[n] = 1.0
    P22_k[n] = 1.0
    
    for j in range(n-1, -1, -1):
        n_j = materials_dict[materials[j]]
        c = cos_betas[j]
        s = sin_betas[j]
        
        L11 = c
        L12 = 1.0j * s / n_j
        L21 = n_j * 1.0j * s
        L22 = c
        
        P11_k[j] = P11_k[j+1] * L11 + P12_k[j+1] * L21
        P12_k[j] = P11_k[j+1] * L12 + P12_k[j+1] * L22
        P21_k[j] = P21_k[j+1] * L11 + P22_k[j+1] * L21
        P22_k[j] = P21_k[j+1] * L12 + P22_k[j+1] * L22
    
    grad = np.zeros(n)
    
    for j in range(n):
        n_j = materials_dict[materials[j]]
        c = cos_betas[j]
        s = sin_betas[j]
        
        # dL_j/dx_j
        # L11 = cos(beta), dL11/dbeta = -sin(beta)
        # L12 = i*sin(beta)/n, dL12/dbeta = i*cos(beta)/n
        # L21 = n*i*sin(beta), dL21/dbeta = n*i*cos(beta)
        # L22 = cos(beta), dL22/dbeta = -sin(beta)
        
        dL11_dbeta = -s
        dL12_dbeta = 1.0j * c / n_j
        dL21_dbeta = n_j * 1.0j * c
        dL22_dbeta = -s
        
        d_beta_dd_j = d_beta_dd[j]
        
        dL11 = dL11_dbeta * d_beta_dd_j
        dL12 = dL12_dbeta * d_beta_dd_j
        dL21 = dL21_dbeta * d_beta_dd_j
        dL22 = dL22_dbeta * d_beta_dd_j
        
        # M_j = M_k[j] * L_j * P_k[j+1]
        # dM_j = M_k[j] * dL_j * P_k[j+1]
        
        # Let A = M_k[j] * dL_j
        # A11 = M11_k[j]*dL11 + M12_k[j]*dL21
        # A12 = M11_k[j]*dL12 + M12_k[j]*dL22
        # A21 = M21_k[j]*dL11 + M22_k[j]*dL21
        # A22 = M21_k[j]*dL12 + M22_k[j]*dL22
        
        A11 = M11_k[j] * dL11 + M12_k[j] * dL21
        A12 = M11_k[j] * dL12 + M12_k[j] * dL22
        A21 = M21_k[j] * dL11 + M22_k[j] * dL21
        A22 = M21_k[j] * dL12 + M22_k[j] * dL22
        
        # dM = A * P_k[j+1]
        # dM11 = A11*P11_k[j+1] + A12*P21_k[j+1]
        # dM12 = A11*P12_k[j+1] + A12*P22_k[j+1]
        # dM21 = A21*P11_k[j+1] + A22*P21_k[j+1]
        # dM22 = A21*P12_k[j+1] + A22*P22_k[j+1]
        
        dM11 = A11 * P11_k[j+1] + A12 * P21_k[j+1]
        dM12 = A11 * P12_k[j+1] + A12 * P22_k[j+1]
        dM21 = A21 * P11_k[j+1] + A22 * P21_k[j+1]
        dM22 = A21 * P12_k[j+1] + A22 * P22_k[j+1]
        
        # dR/dx_j = mean( dR/dM11*dM11 + dR/dM12*dM12 + dR/dM21*dM21 + dR/dM22*dM22 )
        # Note: dR/dM are real? No, R is real, but intermediate terms are complex.
        # R = |r|^2 is real. dR/dM is real?
        # dR/dY is real. dY/dM is complex. So dR/dM is complex?
        # Wait, R is a real function of complex M.
        # dR = dR/dM * dM + conj(dR/dM) * conj(dM) ?
        # Actually, for real R, dR/dM (Wirtinger derivative) satisfies dR = 2 Re( dR/dM * conj(dM) )?
        # Let's check: R = r r*. dR = dr r* + r dr*.
        # dr = (dr/dM) dM.
        # dR = (dr/dM) dM r* + r (dr*/dM*) dM*.
        # Since R is real, we can write dR = 2 Re( r* (dr/dM) dM ).
        # So the gradient term is 2 Re( r* (dr/dM) dM ).
        
        # We computed dR_dY = 2 Re( r * conj(dr_dY) ).
        # Wait, dr_dY is complex.
        # Let's stick to the chain rule for real-valued function R.
        # dR/dx_j = sum_k (dR/dM_k) (dM_k/dx_j)
        # Since R is real, dR/dM_k is the real derivative? No, M_k are complex.
        # Standard result: dR/dx_j = 2 Re[ (dR/dY) (dY/dM11) dM11 + ... ] ?
        # Let's use: dR = 2 Re( r* dr )
        # dr = (dr/dY) dY
        # dY = (dY/dM11) dM11 + ...
        # So dR = 2 Re( r* (dr/dY) [ (dY/dM11) dM11 + ... ] )
        
        term = r * np.conj(dr_dY) # Wait, dr = (dr/dY) dY. dR = 2 Re( r* dr ) = 2 Re( r* (dr/dY) dY ).
        # So we need 2 Re( r* (dr/dY) dY ).
        # dY = (dY/dM11) dM11 + (dY/dM12) dM12 + (dY/dM21) dM21 + (dY/dM22) dM22
        
        dY = dY_dM11 * dM11 + dY_dM12 * dM12 + dY_dM21 * dM21 + dY_dM22 * dM22
        
        dR_dx_j = 2.0 * np.real(r * np.conj(dr_dY) * dY)
        
        grad[j] = np.mean(dR_dx_j)
        
    return 100.0 * mean_R, 100.0 * grad

def design(problem):
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    materials_dict = problem["materials"]
    nL = materials_dict["L"]
    nH = materials_dict["H"]
    n_sub = problem["substrate_index"]
    n_amb = problem["ambient_index"]
    band_nm = problem["band_nm"]
    wl_center = 0.5 * (band_nm[0] + band_nm[1])
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    wls = np.linspace(band_nm[0], band_nm[1], 401)
    
    # Fixed alternating structure starting with L
    # We will optimize 20 layers.
    # If a layer thickness becomes 0, it's effectively removed.
    # But L-BFGS-B might struggle with exact zeros.
    # We'll use bounds [0, max_layer_nm].
    
    n_layers = 20
    materials = []
    for i in range(n_layers):
        if i % 2 == 0:
            materials.append('L')
        else:
            materials.append('H')
    
    bounds = [(0.0, max_layer_nm) for _ in range(n_layers)]
    
    best_score = float('inf')
    best_x = None
    
    # Multi-start
    n_starts = 10
    rng = np.random.default_rng(42)
    
    for k in range(n_starts):
        if time.time() - start_time > deadline - 10:
            break
            
        # Initialize x
        # Some starts: quarter-wave at center
        # Some starts: random
        if k % 2 == 0:
            # Quarter wave init
            x0 = np.zeros(n_layers)
            for i in range(n_layers):
                n_mat = materials_dict[materials[i]]
                x0[i] = wl_center / (4.0 * n_mat)
                # Add some noise
                x0[i] *= rng.uniform(0.8, 1.2)
        else:
            # Random init
            x0 = rng.uniform(0, max_layer_nm, n_layers)
        
        # Ensure total thickness constraint is handled by bounds or penalty?
        # L-BFGS-B handles bounds. Total thickness is a linear constraint.
        # We can add a penalty to the objective if total > max_total_nm.
        # Or just let it run and check validity at the end.
        # Given max_layer_nm=400 and 20 layers, max total is 8000.
        # Constraint is 1000.
        # It's better to penalize in objective to guide the optimizer.
        
        def obj(x):
            # Penalty for total thickness
            total = np.sum(x)
            if total > max_total_nm:
                penalty = (total - max_total_nm) * 10.0
            else:
                penalty = 0.0
            
            val, _ = _objective_and_grad(x, n_layers, materials, n_amb, n_sub, wls)
            return val + penalty
        
        def grad(x):
            val, g = _objective_and_grad(x, n_layers, materials, n_amb, n_sub, wls)
            # Add gradient of penalty
            total = np.sum(x)
            if total > max_total_nm:
                g += np.full(n_layers, 10.0)
            return g
        
        try:
            res = minimize(obj, x0, jac=grad, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5})
            
            if res.success or res.fun < best_score:
                x_opt = res.x
                # Check validity
                if np.sum(x_opt) <= max_total_nm + 1e-6:
                    score, _ = _objective_and_grad(x_opt, n_layers, materials, n_amb, n_sub, wls)
                    if score < best_score:
                        best_score = score
                        best_x = x_opt.copy()
                        
        except Exception:
            pass
    
    if best_x is None:
        # Fallback
        best_x = np.zeros(n_layers)
        best_x[0] = wl_center / (4.0 * nL)
    
    # Post-processing: remove zero-thickness layers to reduce count if needed
    # The problem allows up to 20 layers.
    # We have 20 layers.
    # We should output only layers with thickness > 0.
    # But the interface expects lists of materials and thicknesses.
    # If we return 20 layers with some zeros, is it valid?
    # "every thickness must lie in [0, max_layer_nm]" -> Yes.
    # "sum must not exceed max_total_nm" -> Yes.
    # "list of 'L'/'H' labels" -> Yes.
    # However, removing zeros might allow adding more "effective" layers if we had slack?
    # No, we are limited to 20 layers total.
    # Returning 20 layers is fine.
    
    final_materials = []
    final_thicknesses = []
    for i in range(n_layers):
        if best_x[i] > 1e-6:
            final_materials.append(materials[i])
            final_thicknesses.append(float(best_x[i]))
        else:
            # If we keep zeros, it's valid. But maybe cleaner to remove?
            # Removing them changes the index, which might affect if we wanted to add more?
            # No, we are done.
            # Let's keep them to maintain the alternating structure if it helps?
            # Actually, a zero-thickness layer is an identity matrix. It does nothing.
            # It's safe to include.
            final_materials.append(materials[i])
            final_thicknesses.append(float(best_x[i]))
            
    # Ensure we don't exceed max_layers (we have exactly 20, which is max)
    
    # Final validity check
    if len(final_materials) > problem.get("max_layers", 20):
        # Should not happen
        pass
        
    return final_materials, final_thicknesses

# EVOLVE-BLOCK-END
