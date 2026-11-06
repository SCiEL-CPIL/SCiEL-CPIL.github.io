"""
Broadband antireflection coating design using differentiable optimization.
Strategy:
1. Fix the layer sequence to an alternating L-H pattern (20 layers).
2. Implement the characteristic matrix method with automatic differentiation (custom gradients).
3. Use L-BFGS-B (via scipy.optimize.minimize) to optimize the physical thicknesses.
4. This allows finding non-QWOT (quarter-wave optical thickness) solutions that 
   minimize mean reflectance more effectively than discrete search + local polish.
"""
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _reflectance_and_grad(materials, thicknesses, wls, ambient_index=1.0, substrate_index=1.52, materials_dict=None):
    """
    Compute mean reflectance and its gradient with respect to thicknesses.
    Uses forward-mode automatic differentiation manually implemented for the matrix chain.
    
    materials: list of 'L'/'H'
    thicknesses: list of floats (nm)
    wls: array of wavelengths
    Returns: (mean_reflectance_percent, gradient_array)
    """
    if materials_dict is None:
        materials_dict = {'L': 1.38, 'H': 2.35}
    
    n_layers = len(materials)
    if n_layers == 0:
        r = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return 100.0 * r*r, np.zeros_like(thicknesses)
        
    # We will propagate the total matrix M and its derivatives dM/dt_i
    # M = M_N ... M_1
    # dM/dt_k = M_N ... M_{k+1} * dM_k/dt_k * M_{k-1} ... M_1
    
    # Initialize matrices
    M11 = np.ones_like(wls, dtype=complex)
    M12 = np.zeros_like(wls, dtype=complex)
    M21 = np.zeros_like(wls, dtype=complex)
    M22 = np.ones_like(wls, dtype=complex)
    
    # Store layer matrices and their derivatives for backpropagation
    # L_i matrices
    L11_list = []
    L12_list = []
    L21_list = []
    L22_list = []
    
    # Derivatives of L_i w.r.t t_i
    dL11_dt_list = []
    dL12_dt_list = []
    dL21_dt_list = []
    dL22_dt_list = []
    
    # Pre-compute layer parameters
    for i, (mat, t) in enumerate(zip(materials, thicknesses)):
        n = materials_dict[mat]
        beta = 2.0 * np.pi * n * t / wls
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        L11 = cos_b
        L12 = 1.0j * sin_b / n
        L21 = n * 1.0j * sin_b
        L22 = cos_b
        
        L11_list.append(L11)
        L12_list.append(L12)
        L21_list.append(L21)
        L22_list.append(L22)
        
        # Derivatives w.r.t t
        # dbeta/dt = 2*pi*n/wls
        dbeta = 2.0 * np.pi * n / wls
        
        dL11 = -sin_b * dbeta
        dL12 = 1.0j * cos_b * dbeta / n
        dL21 = n * 1.0j * cos_b * dbeta
        dL22 = -sin_b * dbeta
        
        dL11_dt_list.append(dL11)
        dL12_dt_list.append(dL12)
        dL21_dt_list.append(dL21)
        dL22_dt_list.append(dL22)
        
    # Forward pass: Compute M_total
    # M_total = M_N @ ... @ M_1
    # We need M_total and the "prefix" products P_k = M_k @ ... @ M_1 
    # and "suffix" products S_k = M_N @ ... @ M_{k+1}
    
    # Let's compute all prefix products P_k = M_k ... M_1
    # P_0 = I
    # P_k = M_k @ P_{k-1}
    
    # And suffix products S_k = M_N ... M_{k+1}
    # S_N = I
    # S_k = M_{k+1} @ S_{k+1} ? No, S_k is the product of layers above k.
    # M_total = S_1 @ P_N ? No.
    # M_total = M_N @ ... @ M_1.
    # dM_total/dt_k = (M_N @ ... @ M_{k+1}) @ (dM_k/dt_k) @ (M_{k-1} @ ... @ M_1)
    # Let Right_k = M_{k-1} @ ... @ M_1 (Prefix product up to k-1)
    # Let Left_k = M_N @ ... @ M_{k+1} (Suffix product from k+1 to N)
    
    # Compute Prefix products P_k = M_k @ ... @ M_1
    # P_k[0] = M_1
    # P_k[k] = M_{k+1} @ P_k[k-1]
    
    P11 = np.zeros((n_layers, len(wls)), dtype=complex)
    P12 = np.zeros((n_layers, len(wls)), dtype=complex)
    P21 = np.zeros((n_layers, len(wls)), dtype=complex)
    P22 = np.zeros((n_layers, len(wls)), dtype=complex)
    
    # P_0 is Identity, but we start storing from index 0 (layer 0)
    # P_0 (for layer 0) = M_0
    P11[0] = L11_list[0]
    P12[0] = L12_list[0]
    P21[0] = L21_list[0]
    P22[0] = L22_list[0]
    
    for k in range(1, n_layers):
        # P_k = M_k @ P_{k-1}
        # M_k is L_k
        # P_{k-1} is stored in P*_k-1
        # Note: Matrix multiplication A @ B
        # A11 B11 + A12 B21
        P11[k] = L11_list[k] * P11[k-1] + L12_list[k] * P21[k-1]
        P12[k] = L11_list[k] * P12[k-1] + L12_list[k] * P22[k-1]
        P21[k] = L21_list[k] * P11[k-1] + L22_list[k] * P21[k-1]
        P22[k] = L21_list[k] * P12[k-1] + L22_list[k] * P22[k-1]
        
    # M_total = P_{N-1} (if 0-indexed, last is N-1)
    M11_total = P11[n_layers-1]
    M12_total = P12[n_layers-1]
    M21_total = P21[n_layers-1]
    M22_total = P22[n_layers-1]
    
    # Compute Suffix products S_k = M_N @ ... @ M_{k+1}
    # S_{N-1} (last layer) = I
    # S_k = M_{k+1} @ S_{k+1}
    
    S11 = np.zeros((n_layers, len(wls)), dtype=complex)
    S12 = np.zeros((n_layers, len(wls)), dtype=complex)
    S21 = np.zeros((n_layers, len(wls)), dtype=complex)
    S22 = np.zeros((n_layers, len(wls)), dtype=complex)
    
    # S for last layer (index n_layers-1) is Identity
    S11[n_layers-1] = 1.0
    S12[n_layers-1] = 0.0
    S21[n_layers-1] = 0.0
    S22[n_layers-1] = 1.0
    
    for k in range(n_layers-2, -1, -1):
        # S_k = M_{k+1} @ S_{k+1}
        # M_{k+1} is L_{k+1}
        S11[k] = L11_list[k+1] * S11[k+1] + L12_list[k+1] * S21[k+1]
        S12[k] = L11_list[k+1] * S12[k+1] + L12_list[k+1] * S22[k+1]
        S21[k] = L21_list[k+1] * S11[k+1] + L22_list[k+1] * S21[k+1]
        S22[k] = L21_list[k+1] * S12[k+1] + L22_list[k+1] * S22[k+1]
        
    # Now compute Y and R
    Y_s = substrate_index
    Y_0 = ambient_index
    
    # Y = (M21 + M22 * Y_s) / (M11 + M12 * Y_s)
    num = M21_total + M22_total * Y_s
    den = M11_total + M12_total * Y_s
    Y = num / den
    
    r = (Y_0 - Y) / (Y_0 + Y)
    R = np.abs(r)**2
    
    mean_R = np.mean(R)
    mean_R_percent = 100.0 * mean_R
    
    # Gradient Calculation
    # dR/dt_k = sum_wl dR_wl / dM_total @ dM_total/dt_k
    # This is complex. Let's use the chain rule.
    # R = |r|^2 = r * r*
    # dR = r* dr + r dr*
    
    # dr = d/dY [ (Y0 - Y)/(Y0 + Y) ] dY
    # dr = -2 Y0 / (Y0 + Y)^2 dY
    
    # dY = (dNum * Den - Num * dDen) / Den^2
    
    # We need dM_total/dt_k.
    # dM_total/dt_k = S_k @ dL_k/dt @ P_{k-1} (where P_{-1}=I)
    
    grad = np.zeros(n_layers)
    
    # Precompute dr/dY
    dY = np.zeros_like(Y)
    # We need dM_total/dt_k for each k to get dY_k
    
    # Let's compute dM_total/dt_k for all k
    # dM11_k = S11_k * dL11_k * P11_{k-1} + ... 
    # This is 4 matrix multiplications per k. 
    # Since n_layers is small (20) and wls is 401, this is feasible.
    
    # To save time, we can compute the gradient of the objective w.r.t M_total elements first?
    # No, M_total depends on t_k.
    
    # Let's iterate k
    for k in range(n_layers):
        # Get S_k
        S11_k = S11[k]
        S12_k = S12[k]
        S21_k = S21[k]
        S22_k = S22[k]
        
        # Get dL_k/dt_k
        dL11_k = dL11_dt_list[k]
        dL12_k = dL12_dt_list[k]
        dL21_k = dL21_dt_list[k]
        dL22_k = dL22_dt_list[k]
        
        # Get P_{k-1}. If k==0, P_{-1} = I
        if k == 0:
            P11_km1 = np.ones_like(wls, dtype=complex)
            P12_km1 = np.zeros_like(wls, dtype=complex)
            P21_km1 = np.zeros_like(wls, dtype=complex)
            P22_km1 = np.ones_like(wls, dtype=complex)
        else:
            P11_km1 = P11[k-1]
            P12_km1 = P12[k-1]
            P21_km1 = P21[k-1]
            P22_km1 = P22[k-1]
            
        # Compute dM_k = S_k @ dL_k @ P_{k-1}
        # First: A = S_k @ dL_k
        A11 = S11_k * dL11_k + S12_k * dL21_k
        A12 = S11_k * dL12_k + S12_k * dL22_k
        A21 = S21_k * dL11_k + S22_k * dL21_k
        A22 = S21_k * dL12_k + S22_k * dL22_k
        
        # Then: dM_k = A @ P_{k-1}
        dM11 = A11 * P11_km1 + A12 * P21_km1
        dM12 = A11 * P12_km1 + A12 * P22_km1
        dM21 = A21 * P11_km1 + A22 * P21_km1
        dM22 = A21 * P12_km1 + A22 * P22_km1
        
        # Now compute dY_k
        # Y = Num / Den
        # dNum = dM21 + dM22 * Y_s
        # dDen = dM11 + dM12 * Y_s
        dNum = dM21 + dM22 * Y_s
        dDen = dM11 + dM12 * Y_s
        
        # dY = (dNum * Den - Num * dDen) / Den^2
        dY_k = (dNum * den - num * dDen) / (den * den)
        
        # dr/dY = -2 Y0 / (Y0 + Y)^2
        dY0_Y = Y_0 + Y
        dr_dY = -2.0 * Y_0 / (dY0_Y * dY0_Y)
        
        dr_k = dr_dY * dY_k
        
        # dR = r* dr + r dr*
        # R = r r*
        # dR = r* dr + r dr*
        # Note: r is complex. dr is complex.
        # dR is real.
        dR_k = np.conj(r) * dr_k + r * np.conj(dr_k)
        
        # Mean dR
        mean_dR_k = np.mean(dR_k).real
        
        grad[k] = 100.0 * mean_dR_k
        
    return mean_R_percent, grad

def design(problem):
    """
    Design using differentiable optimization of thicknesses for a fixed alternating stack.
    """
    start_time = time.time()
    deadline = 145.0 # Safety margin for 150s limit
    
    materials_dict = problem["materials"]
    nL = materials_dict["L"]
    nH = materials_dict["H"]
    n_sub = problem["substrate_index"]
    n_amb = problem["ambient_index"]
    band_nm = problem["band_nm"]
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem.get("max_total_nm", 1000)
    
    wls = np.linspace(band_nm[0], band_nm[1], 401)
    
    # Define a fixed sequence of 20 layers: L H L H ...
    # This ensures a broadband structure.
    n_layers = 20
    materials = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
    
    # Initial guess: QWOT at center wavelength
    wl_center = 0.5 * (band_nm[0] + band_nm[1])
    q_l = wl_center / (4.0 * nL)
    q_h = wl_center / (4.0 * nH)
    
    x0 = np.array([q_l if m == 'L' else q_h for m in materials])
    
    # Constraints:
    # 1. 0 <= t_i <= max_layer_nm
    # 2. sum(t_i) <= max_total_nm
    
    # Objective function for scipy
    def objective_and_grad(x):
        # Handle bounds violations with penalty
        # If any t < 0 or t > max_layer_nm, or sum > max_total_nm, return large value
        # For gradient, we just return the physical gradient; scipy handles bounds.
        # But we must ensure x is within bounds for the physics to be valid (t>0).
        # scipy L-BFGS-B respects bounds.
        
        # Compute reflectance and gradient
        try:
            val, grad = _reflectance_and_grad(materials, x, wls, n_amb, n_sub, materials_dict)
            return val, grad
        except Exception:
            return 1e10, np.zeros_like(x)
    
    # Bounds
    bounds = [(0.0, max_layer_nm) for _ in range(n_layers)]
    
    # Initial result
    best_x = x0.copy()
    best_val = objective_and_grad(x0)[0]
    
    # Run L-BFGS-B
    # Use a wrapper to track time
    def time_limited_callback(xk):
        # Check time
        if time.time() - start_time > deadline - 10:
            raise TimeoutError()
        # Optionally check constraint violation? L-BFGS-B handles bounds.
        # We also have the sum constraint. L-BFGS-B doesn't handle linear equality/inequality constraints natively in this simple way?
        # Actually, scipy.optimize.minimize with method='L-BFGS-B' only handles bounds.
        # We need to handle sum(t) <= max_total_nm.
        # If sum(t) > max_total_nm, we should penalize.
        # Let's add a penalty to the objective if sum > max_total_nm.
        pass
        
    # Modified objective with sum penalty
    def objective_with_penalty(x):
        val, grad = objective_and_grad(x)
        # Sum constraint
        total = np.sum(x)
        if total > max_total_nm:
            # Add penalty
            penalty = 1000.0 * (total - max_total_nm)
            # Gradient of penalty
            grad_penalty = np.full_like(x, 1000.0)
            val += penalty
            grad += grad_penalty
        return val, grad

    # Multi-start
    # Start 1: QWOT
    starts = [x0]
    
    # Start 2: Random valid
    for i in range(5):
        x_rand = np.random.uniform(0, max_layer_nm, n_layers)
        # Scale to fit total constraint
        total = np.sum(x_rand)
        if total > max_total_nm:
            x_rand = x_rand * (max_total_nm / total)
        starts.append(x_rand)
        
    # Start 3: Slightly perturbed QWOT
    for i in range(5):
        x_pert = x0 * np.random.uniform(0.5, 1.5, n_layers)
        x_pert = np.clip(x_pert, 0, max_layer_nm)
        total = np.sum(x_pert)
        if total > max_total_nm:
            x_pert = x_pert * (max_total_nm / total)
        starts.append(x_pert)
        
    best_val = 1e10
    best_x = x0
    
    for start_x in starts:
        if time.time() - start_time > deadline - 20:
            break
            
        try:
            res = minimize(objective_with_penalty, start_x, method='L-BFGS-B', jac=True, bounds=bounds,
                           options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-6, 'maxfun': 5000})
            
            if res.fun < best_val:
                # Check validity
                if np.all(res.x >= 0) and np.all(res.x <= max_layer_nm) and np.sum(res.x) <= max_total_nm + 1e-6:
                    best_val = res.fun
                    best_x = res.x
        except Exception:
            continue
            
    # Final validation and clamping
    final_thicknesses = list(best_x)
    final_materials = materials
    
    # Ensure strict validity
    if not _is_valid(final_materials, final_thicknesses, problem):
        # Fallback to QWOT
        final_materials = ['L']
        final_thicknesses = [q_l]
    else:
        # Clamp tiny negatives
        final_thicknesses = [max(0.0, t) for t in final_thicknesses]
        # Re-check sum
        total = sum(final_thicknesses)
        if total > max_total_nm:
            scale = max_total_nm / total
            final_thicknesses = [t * scale for t in final_thicknesses]
            
    return final_materials, final_thicknesses

def _is_valid(materials, thicknesses, problem):
    """Check if the design satisfies all constraints."""
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    if len(materials) > max_layers:
        return False
    if len(materials) != len(thicknesses):
        return False
    for t in thicknesses:
        if t < -1e-6 or t > max_layer_nm + 1e-6:
            return False
    if sum(thicknesses) > max_total_nm + 1e-6:
        return False
    return True

# EVOLVE-BLOCK-END
