"""Broadband AR coating design: smooth L-BFGS-B optimization of layer stack."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _characteristic_matrix(n, d, wl, lam0):
    """Return 2x2 characteristic matrix for a layer at wavelength wl (nm).
    n: refractive index, d: physical thickness (nm), lam0: reference wavelength."""
    # Optical thickness in units of lam0/4 (so phase = pi*d*n/(2*wl) * (2*pi) ...)
    # Phase delta = (2*pi/lambda) * n * d, with lambda = wl
    delta = (2.0 * math.pi / wl) * n * d
    cos_d = math.cos(delta)
    sin_d = math.sin(delta)
    # Matrix for a non-magnetic medium at normal incidence:
    # M = [[cos(delta), (i/sin(delta))/n], [i*n*sin(delta), cos(delta)]]
    # For computation we work with real/imag or just use the standard form.
    # Standard characteristic matrix (complex):
    #   M11 = cos(delta)
    #   M12 = 1j / n * sin(delta)
    #   M21 = 1j * n * sin(delta)
    #   M22 = cos(delta)
    m11 = cos_d
    m12 = 1j * sin_d / n
    m21 = 1j * n * sin_d
    m22 = cos_d
    return np.array([[m11, m12], [m21, m22]], dtype=complex)


def _reflectance_stack(materials, thicknesses, n_ambient, n_substrate, wl):
    """Compute reflectance (0..1) for a stack at wavelength wl using characteristic matrix method."""
    # Total matrix = product of layer matrices (from air side to substrate)
    total = np.array([[1.0+0j, 0.0+0j], [0.0+0j, 1.0+0j]], dtype=complex)
    for mat_label, d in zip(materials, thicknesses):
        n = n_ambient if mat_label == 'A' else (n_substrate if mat_label == 'S' else materials[mat_label])
        M = _characteristic_matrix(n, d, wl, 1.0)
        total = total @ M
    # For normal incidence:
    # B11 = total[0,0], B12 = total[0,1], B21 = total[1,0], B22 = total[1,1]
    B11 = total[0, 0]
    B12 = total[0, 1]
    B21 = total[1, 0]
    B22 = total[1, 1]
    # Input admittance Y = (n_sub * B21) / (B11 + n_sub * B12) ... 
    # Actually: Y = (B21 + B22 * Y_sub) / (B11 + B12 * Y_sub)
    # where Y_sub = n_substrate (normalized admittance)
    Y_sub = n_substrate
    Y = (B21 + B22 * Y_sub) / (B11 + B12 * Y_sub)
    # Reflectance amplitude: r = (Y - n_ambient) / (Y + n_ambient)
    r = (Y - n_ambient) / (Y + n_ambient)
    R = (r * r.conjugate()).real
    R = max(0.0, min(1.0, R))
    return R


def _evaluate_stack(materials, thicknesses, problem):
    """Compute mean reflectance in percent over the band."""
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    mats = problem["materials"]
    wl0, wl1 = problem["band_nm"]
    n_points = 401
    wls = np.linspace(wl0, wl1, n_points)
    total_R = 0.0
    for wl in wls:
        R = _reflectance_stack(materials, thicknesses, n_ambient, n_substrate, float(wl))
        total_R += R
    mean_R_pct = (total_R / n_points) * 100.0
    return mean_R_pct


def _fast_reflectance_vectorized(materials, thicknesses, n_ambient, n_substrate, wls):
    """Vectorized reflectance computation for all wavelengths at once.
    Returns array of reflectance values (0..1)."""
    n_layers = len(materials)
    if n_layers == 0:
        # Bare substrate
        r = (n_substrate - n_ambient) / (n_substrate + n_ambient)
        return np.full(len(wls), r * r)
    
    # Precompute per-layer parameters
    ns = np.array([mats[mat] for mat in materials], dtype=float)
    ds = np.array(thicknesses, dtype=float)
    
    # Phase for each layer at each wavelength: delta = 2*pi*n*d/wl
    # Shape: (n_wls, n_layers)
    deltas = (2.0 * np.pi * np.outer(wls, 1.0) * 0 + 0)  # placeholder
    # Actually: delta[i,j] = 2*pi*n_j*d_j / wl_i
    deltas = 2.0 * np.pi * ns[None, :] * ds[None, :] / wls[:, None]
    
    cos_d = np.cos(deltas)  # (n_wls, n_layers)
    sin_d = np.sin(deltas)  # (n_wls, n_layers)
    
    # Build total matrix by sequential multiplication
    # Start with identity for each wavelength
    total_m11 = np.ones(len(wls), dtype=complex)
    total_m12 = np.zeros(len(wls), dtype=complex)
    total_m21 = np.zeros(len(wls), dtype=complex)
    total_m22 = np.ones(len(wls), dtype=complex)
    
    for j in range(n_layers):
        n_j = ns[j]
        # Layer matrix elements (vectors over wavelengths)
        m11_j = cos_d[:, j]
        m12_j = 1j * sin_d[:, j] / n_j
        m21_j = 1j * n_j * sin_d[:, j]
        m22_j = cos_d[:, j]
        
        # total = total @ M_j
        new_m11 = total_m11 * m11_j + total_m12 * m21_j
        new_m12 = total_m11 * m12_j + total_m12 * m22_j
        new_m21 = total_m21 * m11_j + total_m22 * m21_j
        new_m22 = total_m21 * m12_j + total_m22 * m22_j
        
        total_m11 = new_m11
        total_m12 = new_m12
        total_m21 = new_m21
        total_m22 = new_m22
    
    # Compute reflectance
    Y_sub = n_substrate
    Y = (total_m21 + total_m22 * Y_sub) / (total_m11 + total_m12 * Y_sub)
    r = (Y - n_ambient) / (Y + n_ambient)
    R = (r * r.conjugate()).real
    R = np.clip(R, 0.0, 1.0)
    return R


def design(problem):
    """Design a broadband AR coating using smooth optimization."""
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s kill
    
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    mats = problem["materials"]
    nL = mats["L"]
    nH = mats["H"]
    wl0, wl1 = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    # Use a reduced set of wavelengths for optimization (faster)
    n_opt_wls = 81
    wls_opt = np.linspace(wl0, wl1, n_opt_wls)
    
    def objective(x):
        """x: vector of [n_layers, thickness_1, ..., thickness_n]
        First element is n_layers (integer-like, we'll round), rest are thicknesses."""
        n_layers = int(round(x[0]))
        n_layers = max(1, min(n_layers, max_layers))
        thicknesses = x[1:1+n_layers]
        # Clip thicknesses to valid range
        thicknesses = np.clip(thicknesses, 0.0, max_layer_nm)
        # Check total thickness constraint (soft penalty)
        total_t = np.sum(thicknesses)
        if total_t > max_total_nm:
            penalty = (total_t - max_total_nm) / max_total_nm * 10.0
        else:
            penalty = 0.0
        
        # Build materials pattern: alternate L, H starting with L
        # For n_layers, use pattern that makes sense for AR coating
        # Standard: start with L (low index) to match air, end with H (high index) to match substrate
        # Pattern: L H L H ... 
        material_labels = []
        for i in range(n_layers):
            if i % 2 == 0:
                material_labels.append('L')
            else:
                material_labels.append('H')
        
        R = _fast_reflectance_vectorized(material_labels, thicknesses, n_ambient, n_substrate, wls_opt)
        mean_R = np.mean(R)
        return mean_R + penalty
    
    # Try multiple initializations and pick the best
    best_score = float('inf')
    best_materials = None
    best_thicknesses = None
    
    # Initial guesses: quarter-wave stack patterns
    centre_wl = 0.5 * (wl0 + wl1)
    qw_L = centre_wl / (4.0 * nL)
    qw_H = centre_wl / (4.0 * nH)
    
    # Multiple starting points
    initializations = []
    
    # 1. Single quarter-wave L
    initializations.append([1.0, qw_L])
    
    # 2. Two layers: L H
    initializations.append([2.0, qw_L, qw_H])
    
    # 3. Three layers: L H L
    initializations.append([3.0, qw_L, qw_H, qw_L])
    
    # 4. Four layers: L H L H
    initializations.append([4.0, qw_L, qw_H, qw_L, qw_H])
    
    # 5. Five layers: L H L H L
    initializations.append([5.0, qw_L, qw_H, qw_L, qw_H, qw_L])
    
    # 6. Six layers
    initializations.append([6.0, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H])
    
    # 7. Random perturbations of quarter-wave
    for _ in range(5):
        n_init = np.random.randint(3, 8)
        x0 = [float(n_init)]
        for i in range(n_init):
            if i % 2 == 0:
                x0.append(qw_L * np.random.uniform(0.5, 1.5))
            else:
                x0.append(qw_H * np.random.uniform(0.5, 1.5))
        initializations.append(x0)
    
    # 8. Broader initializations with varied thicknesses
    for n_init in [4, 6, 8, 10]:
        x0 = [float(n_init)]
        for i in range(n_init):
            if i % 2 == 0:
                x0.append(qw_L * np.random.uniform(0.7, 1.3))
            else:
                x0.append(qw_H * np.random.uniform(0.7, 1.3))
        initializations.append(x0)
    
    # Bounds for optimization
    # x[0] = n_layers (1 to max_layers)
    # x[1:] = thicknesses (0 to max_layer_nm)
    
    for x0 in initializations:
        if time.time() > deadline - 5.0:
            break
        
        n_params = len(x0)
        n_layers_init = int(round(x0[0]))
        n_layers_init = max(1, min(n_layers_init, max_layers))
        
        bounds = [(1.0, float(max_layers))]
        for i in range(n_layers_init):
            bounds.append((0.0, max_layer_nm))
        
        # Pad x0 if needed
        x0_arr = np.array(x0[:1+n_layers_init], dtype=float)
        if len(x0_arr) < 1 + n_layers_init:
            pad = np.zeros(1 + n_layers_init - len(x0_arr))
            x0_arr = np.concatenate([x0_arr, pad])
        
        # Use L-BFGS-B
        try:
            from scipy.optimize import minimize
            result = minimize(
                objective,
                x0_arr,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.success or result.fun < best_score:
                x_opt = result.x
                n_layers_opt = int(round(x_opt[0]))
                n_layers_opt = max(1, min(n_layers_opt, max_layers))
                thicknesses_opt = np.clip(x_opt[1:1+n_layers_opt], 0.0, max_layer_nm)
                
                # Check total thickness
                if np.sum(thicknesses_opt) > max_total_nm:
                    # Scale down
                    scale = max_total_nm / np.sum(thicknesses_opt)
                    thicknesses_opt = thicknesses_opt * scale
                
                material_labels = []
                for i in range(n_layers_opt):
                    if i % 2 == 0:
                        material_labels.append('L')
                    else:
                        material_labels.append('H')
                
                # Evaluate on full 401-point grid
                score = _evaluate_stack(material_labels, thicknesses_opt.tolist(), problem)
                if score < best_score:
                    best_score = score
                    best_materials = material_labels
                    best_thicknesses = thicknesses_opt.tolist()
        except Exception:
            pass
    
    # If no valid result, fall back to simple quarter-wave
    if best_materials is None:
        centre = 0.5 * (wl0 + wl1)
        best_materials = ["L"]
        best_thicknesses = [centre / (4.0 * nL)]
        best_score = _evaluate_stack(best_materials, best_thicknesses, problem)
    
    # Final polish: try a few more local optimizations from the best solution
    if best_materials is not None and time.time() < deadline - 10.0:
        n_best = len(best_materials)
        x_best = [float(n_best)] + list(best_thicknesses)
        
        # Try perturbing and re-optimizing
        for _ in range(3):
            if time.time() > deadline - 5.0:
                break
            x_perturbed = x_best.copy()
            # Small random perturbation
            for i in range(1, len(x_perturbed)):
                x_perturbed[i] += np.random.normal(0, 10.0)
                x_perturbed[i] = max(0.0, min(x_perturbed[i], max_layer_nm))
            
            n_layers_p = int(round(x_perturbed[0]))
            n_layers_p = max(1, min(n_layers_p, max_layers))
            bounds = [(1.0, float(max_layers))]
            for i in range(n_layers_p):
                bounds.append((0.0, max_layer_nm))
            
            x_pert_arr = np.array(x_perturbed[:1+n_layers_p], dtype=float)
            if len(x_pert_arr) < 1 + n_layers_p:
                pad = np.zeros(1 + n_layers_p - len(x_pert_arr))
                x_pert_arr = np.concatenate([x_pert_arr, pad])
            
            try:
                from scipy.optimize import minimize
                result = minimize(
                    objective,
                    x_pert_arr,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if result.fun < best_score:
                    x_opt = result.x
                    n_layers_opt = int(round(x_opt[0]))
                    n_layers_opt = max(1, min(n_layers_opt, max_layers))
                    thicknesses_opt = np.clip(x_opt[1:1+n_layers_opt], 0.0, max_layer_nm)
                    
                    if np.sum(thicknesses_opt) > max_total_nm:
                        scale = max_total_nm / np.sum(thicknesses_opt)
                        thicknesses_opt = thicknesses_opt * scale
                    
                    material_labels = []
                    for i in range(n_layers_opt):
                        if i % 2 == 0:
                            material_labels.append('L')
                        else:
                            material_labels.append('H')
                    
                    score = _evaluate_stack(material_labels, thicknesses_opt.tolist(), problem)
                    if score < best_score:
                        best_score = score
                        best_materials = material_labels
                        best_thicknesses = thicknesses_opt.tolist()
            except Exception:
                pass
    
    # Ensure validity
    if best_materials is None:
        best_materials = ["L"]
        best_thicknesses = [600.0 / (4.0 * nL)]
    
    # Final validation
    for t in best_thicknesses:
        if t < 0 or t > max_layer_nm:
            best_materials = ["L"]
            best_thicknesses = [600.0 / (4.0 * nL)]
            break
    if sum(best_thicknesses) > max_total_nm:
        scale = max_total_nm / sum(best_thicknesses)
        best_thicknesses = [t * scale for t in best_thicknesses]
    if len(best_materials) > max_layers:
        best_materials = best_materials[:max_layers]
        best_thicknesses = best_thicknesses[:max_layers]
    
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
