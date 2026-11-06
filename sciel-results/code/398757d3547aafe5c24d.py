"""
Broadband antireflection coating design via adjoint-based gradient optimization.

Strategy:
- Represent the coating as a piecewise-constant refractive index profile
- Compute exact gradients of mean reflectance w.r.t. layer thicknesses using
  the adjoint method (reverse-mode automatic differentiation through the
  characteristic matrix chain)
- Use L-BFGS-B from multiple diverse initializations
- Try several layer counts and material sequences
- Enforce constraints: thickness in [0, max_layer_nm], sum <= max_total_nm
"""
import numpy as np
import time
from scipy.optimize import minimize

def _compute_reflectance_and_adjoint(wavelengths_nm, indices, thicknesses_nm,
                                     ambient_idx, substrate_idx):
    """
    Compute reflectance and adjoint gradients w.r.t. thicknesses.
    
    Returns:
        R: array of reflectance (fraction) at each wavelength
        grads: array of shape (n_layers,) with dR_mean/d(thickness_i)
    """
    n_layers = len(indices)
    n_wl = len(wavelengths_nm)
    
    if n_layers == 0:
        r0 = (ambient_idx - substrate_idx) / (ambient_idx + substrate_idx)
        R = np.full(n_wl, r0 * r0)
        return R, np.zeros(n_layers)
    
    k = 2.0 * np.pi / wavelengths_nm
    
    # Forward pass: compute M matrices and store intermediate values
    M00 = np.ones(n_wl, dtype=complex)
    M01 = np.zeros(n_wl, dtype=complex)
    M10 = np.zeros(n_wl, dtype=complex)
    M11 = np.ones(n_wl, dtype=complex)
    
    # Store per-layer B matrices and intermediate M states for adjoint
    B00_list = []
    B01_list = []
    B10_list = []
    B11_list = []
    M00_prev = []
    M01_prev = []
    M10_prev = []
    M11_prev = []
    delta_list = []
    
    for i in range(n_layers):
        M00_prev.append(M00.copy())
        M01_prev.append(M01.copy())
        M10_prev.append(M10.copy())
        M11_prev.append(M11.copy())
        
        n_i = indices[i]
        d_i = thicknesses_nm[i]
        delta = k * n_i * d_i
        delta_list.append(delta)
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        B00 = cos_d
        B01 = 1j * sin_d / n_i
        B10 = 1j * n_i * sin_d
        B11 = cos_d
        
        B00_list.append(B00)
        B01_list.append(B01)
        B10_list.append(B10)
        B11_list.append(B11)
        
        new_M00 = M00 * B00 + M01 * B10
        new_M01 = M00 * B01 + M01 * B11
        new_M10 = M10 * B00 + M11 * B10
        new_M11 = M10 * B01 + M11 * B11
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    Y_out = substrate_idx
    Y_in = (M10 + M11 * Y_out) / (M00 + M01 * Y_out)
    
    r = (ambient_idx - Y_in) / (ambient_idx + Y_in)
    R = np.abs(r) ** 2
    R_mean = np.mean(R)
    
    # Adjoint pass: compute gradients w.r.t. thicknesses
    # We need d(R_mean)/d(delta_i) for each layer i
    # R = |r|^2, r = (n0 - Y_in)/(n0 + Y_in)
    # Y_in = (M10 + M11*n_s)/(M00 + M01*n_s)
    
    # Let's compute dY_in/dM_ij and then dM_ij/d(delta_i)
    denom = M00 + M01 * Y_out
    dYin_dM00 = -(M10 + M11 * Y_out) / denom**2
    dYin_dM01 = -Y_out * (M10 + M11 * Y_out) / denom**2
    dYin_dM10 = 1.0 / denom
    dYin_dM11 = Y_out / denom
    
    dr_dYin = (2.0 * ambient_idx) / (ambient_idx + Y_in)**2
    dR_dYin = 2.0 * r * dr_dYin  # d|r|^2/dY = 2*Re(conj(r)*dr/dY)
    
    # Actually, let's be more careful. R = |r|^2 = r * conj(r)
    # dR/dY_in = conj(r) * dr/dY_in + r * conj(dr/dY_in)
    dr_dYin_val = (2.0 * ambient_idx) / (ambient_idx + Y_in)**2
    dR_dYin = conj(r) * dr_dYin_val + r * conj(dr_dYin_val)
    
    dR_dM00 = dR_dYin * dYin_dM00
    dR_dM01 = dR_dYin * dYin_dM01
    dR_dM10 = dR_dYin * dYin_dM10
    dR_dM11 = dR_dYin * dYin_dM11
    
    grads = np.zeros(n_layers)
    
    # Adjoint: propagate sensitivity back through layers
    # a_M00, a_M01, a_M10, a_M11 are adjoint variables for the final M
    a_M00 = dR_dM00 / n_wl
    a_M01 = dR_dM01 / n_wl
    a_M10 = dR_dM10 / n_wl
    a_M11 = dR_dM11 / n_wl
    
    for i in range(n_layers - 1, -1, -1):
        # M_new = M_prev @ B_i
        # a_M_prev = (a_M_new @ B_i^T)^T = B_i^T @ a_M_new (in matrix sense)
        # But we're dealing with element-wise products since B depends on wavelength
        # M_new[0,0] = M_prev[0,0]*B00 + M_prev[0,1]*B10
        # So a_M_prev[0,0] gets contributions from a_M_new[0,0]*B00 and a_M_new[0,1]*B10
        # etc.
        
        a_prev_M00 = a_M00 * B00_list[i] + a_M10 * B00_list[i]
        a_prev_M01 = a_M00 * B01_list[i] + a_M10 * B01_list[i]
        a_prev_M10 = a_M01 * B10_list[i] + a_M11 * B10_list[i]
        a_prev_M11 = a_M01 * B11_list[i] + a_M11 * B11_list[i]
        
        # Now compute dR_mean/d(delta_i)
        # B00 = cos(delta), dB00/ddelta = -sin(delta)
        # B01 = 1j*sin(delta)/n_i, dB01/ddelta = 1j*cos(delta)/n_i
        # B10 = 1j*n_i*sin(delta), dB10/ddelta = 1j*n_i*cos(delta)
        # B11 = cos(delta), dB11/ddelta = -sin(delta)
        
        delta_i = delta_list[i]
        n_i = indices[i]
        
        sin_d = np.sin(delta_i)
        cos_d = np.cos(delta_i)
        
        # d(B00)/ddelta = -sin_d
        # d(B01)/ddelta = 1j*cos_d/n_i
        # d(B10)/ddelta = 1j*n_i*cos_d
        # d(B11)/ddelta = -sin_d
        
        # The contribution to the gradient from layer i:
        # sum over elements of a_M_prev * M_prev * d(B)/ddelta
        # Actually, the sensitivity is:
        # dR/d(delta_i) = sum of a_M_new[0,0]*M_prev[0,1]*dB10 + a_M_new[0,0]*M_prev[0,0]*dB00 + ...
        
        # Let me redo this more carefully.
        # M_new = M_prev * B
        # dM_new/d(delta_i) = M_prev * dB/d(delta_i)
        # dR/d(delta_i) = trace(a_M_new^T * dM_new/d(delta_i))
        # = a_M_new[0,0] * (M_prev[0,0]*dB00 + M_prev[0,1]*dB10)
        # + a_M_new[0,1] * (M_prev[0,0]*dB01 + M_prev[0,1]*dB11)
        # + a_M_new[1,0] * (M_prev[1,0]*dB00 + M_prev[1,1]*dB10)
        # + a_M_new[1,1] * (M_prev[1,0]*dB01 + M_prev[1,1]*dB11)
        
        M00_p = M00_prev[i]
        M01_p = M01_prev[i]
        M10_p = M10_prev[i]
        M11_p = M11_prev[i]
        
        dB00 = -sin_d
        dB01 = 1j * cos_d / n_i
        dB10 = 1j * n_i * cos_d
        dB11 = -sin_d
        
        grad_delta_i = (
            a_M00 * (M00_p * dB00 + M01_p * dB10) +
            a_M01 * (M00_p * dB01 + M01_p * dB11) +
            a_M10 * (M10_p * dB00 + M11_p * dB10) +
            a_M11 * (M10_p * dB01 + M11_p * dB11)
        )
        
        # grad_delta_i is complex; take real part
        grads[i] = np.real(np.mean(grad_delta_i))
        
        # Update adjoint variables
        a_M00 = a_prev_M00
        a_M01 = a_prev_M01
        a_M10 = a_prev_M10
        a_M11 = a_prev_M11
    
    # Convert dR/d(delta_i) to dR/d(thickness_i)
    # delta_i = k * n_i * thickness_i
    # d(delta_i)/d(thickness_i) = k * n_i
    k_arr = 2.0 * np.pi / wavelengths_nm
    for i in range(n_layers):
        grads[i] *= np.mean(k_arr * indices[i])
    
    return R, grads


def _objective_and_grad(x, indices, wavelengths_nm, ambient_idx, substrate_idx,
                        max_layer_nm, max_total_nm):
    """
    Objective and gradient for L-BFGS-B.
    """
    x = np.asarray(x, dtype=float)
    x_clipped = np.clip(x, 0.0, max_layer_nm)
    
    R, grads = _compute_reflectance_and_adjoint(
        wavelengths_nm, indices, x_clipped, ambient_idx, substrate_idx
    )
    
    # Check total thickness constraint
    total = np.sum(x_clipped)
    if total > max_total_nm + 1e-9:
        return 100.0, np.zeros_like(x)
    
    obj = np.mean(R) * 100.0
    return obj, grads


def _design_for_config(n_layers, labels, materials, ambient_idx, substrate_idx,
                       wavelengths_nm, center_wl, max_total_nm, max_layer_nm,
                       time_deadline):
    """
    Design for a specific layer configuration using L-BFGS-B.
    """
    indices = [materials[label] for label in labels]
    n = n_layers
    
    bounds = [(0.0, float(max_layer_nm))] * n
    
    def f_and_grad(x):
        return _objective_and_grad(x, indices, wavelengths_nm, ambient_idx,
                                   substrate_idx, max_layer_nm, max_total_nm)
    
    def f(x):
        return f_and_grad(x)[0]
    
    # Generate multiple diverse initializations
    initializations = []
    
    # Quarter-wave stack
    x0_qw = np.array([center_wl / (4.0 * n_i) for n_i in indices])
    if np.sum(x0_qw) > max_total_nm:
        x0_qw = x0_qw * (max_total_nm / np.sum(x0_qw))
    x0_qw = np.clip(x0_qw, 0, max_layer_nm)
    initializations.append(x0_qw)
    
    # Half-wave stack
    x0_hw = np.array([center_wl / (2.0 * n_i) for n_i in indices])
    if np.sum(x0_hw) > max_total_nm:
        x0_hw = x0_hw * (max_total_nm / np.sum(x0_hw))
    x0_hw = np.clip(x0_hw, 0, max_layer_nm)
    initializations.append(x0_hw)
    
    # Random initializations
    rng = np.random.RandomState(42)
    for _ in range(3):
        x0_rand = rng.uniform(0, max_layer_nm, size=n)
        if np.sum(x0_rand) > max_total_nm:
            x0_rand = x0_rand * (max_total_nm / np.sum(x0_rand))
        x0_rand = np.clip(x0_rand, 0, max_layer_nm)
        initializations.append(x0_rand)
    
    # Linearly spaced thicknesses
    x0_lin = np.linspace(max_layer_nm * 0.1, max_layer_nm * 0.5, n)
    if np.sum(x0_lin) > max_total_nm:
        x0_lin = x0_lin * (max_total_nm / np.sum(x0_lin))
    x0_lin = np.clip(x0_lin, 0, max_layer_nm)
    initializations.append(x0_lin)
    
    best_x = None
    best_f = np.inf
    
    for x0 in initializations:
        if time.time() > time_deadline:
            break
        
        try:
            result = minimize(
                f, x0, jac=lambda x: f_and_grad(x)[1],
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8,
                         'disp': False}
            )
            
            if result.success or result.fun < best_f:
                x_opt = np.clip(result.x, 0, max_layer_nm)
                if np.sum(x_opt) <= max_total_nm + 1e-9:
                    f_opt = f(x_opt)
                    if f_opt < best_f:
                        best_f = f_opt
                        best_x = x_opt
        except Exception:
            continue
    
    if best_x is None:
        best_x = x0_qw
        best_f = f(best_x)
    
    return labels, best_x, best_f


def design(problem):
    """
    Design a broadband antireflection coating.
    Returns: (labels, thicknesses)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wavelengths_nm = np.linspace(band_nm[0], band_nm[1], 401)
    center_wl = 0.5 * (band_nm[0] + band_nm[1])
    
    best_labels = None
    best_thicknesses = None
    best_score = np.inf
    
    # Try different layer counts and material sequences
    # Focus on odd numbers of layers for good AR performance
    layer_configs = []
    
    # Alternating patterns starting with L or H
    for n_layers in [5, 7, 9, 11, 13]:
        if n_layers > max_layers:
            continue
        for start_char in ['L', 'H']:
            labels = [start_char if i % 2 == 0 else ('H' if start_char == 'L' else 'L')
                      for i in range(n_layers)]
            layer_configs.append((n_layers, labels))
    
    # Some non-alternating patterns
    if max_layers >= 7:
        layer_configs.append((7, ['L', 'L', 'H', 'H', 'L', 'L', 'H']))
        layer_configs.append((7, ['H', 'H', 'L', 'L', 'H', 'H', 'L']))
    
    if max_layers >= 9:
        layer_configs.append((9, ['L', 'L', 'H', 'H', 'L', 'H', 'H', 'L', 'L']))
        layer_configs.append((9, ['H', 'H', 'L', 'L', 'H', 'L', 'L', 'H', 'H']))
    
    # Allocate time per configuration
    n_configs = len(layer_configs)
    time_per_config = max(5.0, (deadline - start_time - 15.0) / max(n_configs, 1))
    
    for n_layers, labels in layer_configs:
        if time.time() > deadline - 15.0:
            break
        
        try:
            labels_opt, thicknesses, score = _design_for_config(
                n_layers, labels, materials, ambient_idx, substrate_idx,
                wavelengths_nm, center_wl, max_total_nm, max_layer_nm,
                min(time.time() + time_per_config, deadline - 15.0)
            )
            
            if score < best_score:
                best_score = score
                best_labels = labels_opt
                best_thicknesses = thicknesses
        except Exception:
            continue
    
    # Fallback if nothing worked
    if best_labels is None:
        best_labels = ['L']
        best_thicknesses = np.array([center_wl / (4.0 * materials['L'])])
    
    # Final constraint check
    thicknesses = np.array(best_thicknesses)
    thicknesses = np.clip(thicknesses, 0, max_layer_nm)
    if np.sum(thicknesses) > max_total_nm:
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
    
    return best_labels, thicknesses.tolist()
