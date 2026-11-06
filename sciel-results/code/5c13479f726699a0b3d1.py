"""
Broadband antireflection coating design via analytic-gradient L-BFGS-B optimization.

Strategy:
- Compute exact analytic gradients of mean reflectance w.r.t. each thickness
  using chain-rule propagation through the characteristic transfer matrix
- Use scipy.optimize.minimize (L-BFGS-B) with analytic gradients for fast convergence
- Search over diverse layer configurations (not just alternating L/H)
- Multi-start from physically-motivated initial designs
"""
import numpy as np
import time
from scipy.optimize import minimize

def _compute_reflectance(wavelengths_nm, indices, thicknesses_nm, ambient_idx, substrate_idx):
    """
    Compute reflectance (fraction, 0-1) at each wavelength using the characteristic matrix method.
    """
    n_layers = len(indices)
    if n_layers == 0:
        r0 = (ambient_idx - substrate_idx) / (ambient_idx + substrate_idx)
        return np.full(len(wavelengths_nm), r0 * r0)
    
    k = 2.0 * np.pi / wavelengths_nm
    
    M00 = np.ones(len(wavelengths_nm), dtype=complex)
    M01 = np.zeros(len(wavelengths_nm), dtype=complex)
    M10 = np.zeros(len(wavelengths_nm), dtype=complex)
    M11 = np.ones(len(wavelengths_nm), dtype=complex)
    
    for i in range(n_layers):
        n_i = indices[i]
        d_i = thicknesses_nm[i]
        delta = k * n_i * d_i
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        B00 = cos_d
        B01 = 1j * sin_d / n_i
        B10 = 1j * n_i * sin_d
        B11 = cos_d
        
        new_M00 = M00 * B00 + M01 * B10
        new_M01 = M00 * B01 + M01 * B11
        new_M10 = M10 * B00 + M11 * B10
        new_M11 = M10 * B01 + M11 * B11
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    Y_out = substrate_idx
    Y_in = (M10 + M11 * Y_out) / (M00 + M01 * Y_out)
    
    r = (ambient_idx - Y_in) / (ambient_idx + Y_in)
    R = np.abs(r) ** 2
    
    return R

def _objective_and_gradient(thicknesses, indices, wavelengths_nm, ambient_idx, substrate_idx,
                            max_layer_nm, max_total_nm):
    """
    Compute mean reflectance in percent AND its analytic gradient w.r.t. thicknesses.
    Uses chain-rule propagation through the transfer matrix.
    """
    thicknesses = np.asarray(thicknesses, dtype=float)
    n_layers = len(indices)
    n_wl = len(wavelengths_nm)
    
    if np.any(thicknesses < -1e-9) or np.any(thicknesses > max_layer_nm + 1e-9):
        return 100.0, np.zeros(n_layers)
    if np.sum(thicknesses) > max_total_nm + 1e-9:
        return 100.0, np.zeros(n_layers)
    
    thicknesses = np.clip(thicknesses, 0.0, max_layer_nm)
    
    k = 2.0 * np.pi / wavelengths_nm
    
    # Forward pass: compute M matrices and store intermediate values for backward pass
    # M_total = M_0 * M_1 * ... * M_{n-1}
    # We need dR/dd_i for each layer i
    
    # Store all B matrices and M products
    B_list = []
    M_list = []  # M_list[i] = product of B_0 * B_1 * ... * B_{i-1} (i.e., M before layer i)
    
    M00 = np.ones(n_wl, dtype=complex)
    M01 = np.zeros(n_wl, dtype=complex)
    M10 = np.zeros(n_wl, dtype=complex)
    M11 = np.ones(n_wl, dtype=complex)
    
    M_list.append((M00.copy(), M01.copy(), M10.copy(), M11.copy()))
    
    for i in range(n_layers):
        n_i = indices[i]
        d_i = thicknesses[i]
        delta = k * n_i * d_i
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        B00 = cos_d
        B01 = 1j * sin_d / n_i
        B10 = 1j * n_i * sin_d
        B11 = cos_d
        
        # Derivatives of B w.r.t. d_i
        dB00_dd = -k * n_i * sin_d
        dB01_dd = 1j * k * n_i * cos_d / n_i  # = 1j * k * cos_d
        dB10_dd = 1j * k * n_i * n_i * cos_d  # = 1j * k * n_i^2 * cos_d
        dB11_dd = -k * n_i * sin_d
        
        B_list.append((B00, B01, B10, B11, dB00_dd, dB01_dd, dB10_dd, dB11_dd))
        
        new_M00 = M00 * B00 + M01 * B10
        new_M01 = M00 * B01 + M01 * B11
        new_M10 = M10 * B00 + M11 * B10
        new_M11 = M10 * B01 + M11 * B11
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    # Y_in = (M10 + M11 * Ys) / (M00 + M01 * Ys)
    Ys = substrate_idx
    num = M10 + M11 * Ys
    den = M00 + M01 * Ys
    Y_in = num / den
    
    r = (ambient_idx - Y_in) / (ambient_idx + Y_in)
    R = np.abs(r) ** 2
    
    mean_R = np.mean(R)
    obj = mean_R * 100.0
    
    # Backward pass: compute dR/dM for the total matrix, then propagate back
    # R = |r|^2 = r * conj(r)
    # dr/dY_in = -2*ambient_idx / (ambient_idx + Y_in)^2
    # dY_in/dnum = 1/den, dY_in/den = -num/den^2
    # dnum/dM10 = 1, dnum/dM11 = Ys
    # dden/dM00 = 1, dden/dM01 = Ys
    
    # dR/dM elements (for the total matrix M_total)
    # R = |r|^2, r = (na - Y)/(na + Y)
    # dr/dY = -2*na/(na+Y)^2
    dY_in = 1.0 / den
    dden = -num / (den * den)
    
    dr_dY = -2.0 * ambient_idx / (ambient_idx + Y_in) ** 2
    
    dR_dnum = 2.0 * np.real(r * dr_dY * dY_in)  # dR/dnum = 2*Re(r * dr/dY * dY/dnum)
    dR_dnum = 2.0 * np.real(r * dr_dY * dY_in)
    dR_dden = 2.0 * np.real(r * dr_dY * dden)
    
    # dR/dM00 = dR_dnum * 0 + dR_dden * 1 = dR_dden
    # dR/dM01 = dR_dnum * 0 + dR_dden * Ys = dR_dden * Ys
    # dR/dM10 = dR_dnum * 1 + dR_dden * 0 = dR_dnum
    # dR/dM11 = dR_dnum * Ys + dR_dden * 0 = dR_dnum * Ys
    
    dR_dM00 = dR_dden
    dR_dM01 = dR_dden * Ys
    dR_dM10 = dR_dnum
    dR_dM11 = dR_dnum * Ys
    
    # Now propagate backward through the matrix product
    # M_total = M_0 * B_0 * B_1 * ... * B_{n-1}
    # Let P_i = B_i * B_{i+1} * ... * B_{n-1} (product from layer i to end)
    # dR/dd_i = sum over matrix elements of dR/dM_total * dM_total/dd_i
    # dM_total/dd_i = (product before i) * dB_i/dd_i * (product after i)
    
    # We'll compute this by forward/backward sweep
    # grad[i] = trace-like contraction of dR_dM with dM_total/dd_i
    
    # Let's define: for layer i, dM_total/dd_i = M_before_i * dB_i/dd_i * M_after_i
    # where M_before_i = B_0 * B_1 * ... * B_{i-1} (stored in M_list[i])
    # and M_after_i = B_{i+1} * ... * B_{n-1}
    
    # We need M_after_i. Let's compute suffix products.
    # suffix[i] = B_i * B_{i+1} * ... * B_{n-1}
    # suffix[n] = I
    # suffix[i] = B_i * suffix[i+1]
    
    suffix_M00 = np.ones(n_wl, dtype=complex)
    suffix_M01 = np.zeros(n_wl, dtype=complex)
    suffix_M10 = np.zeros(n_wl, dtype=complex)
    suffix_M11 = np.ones(n_wl, dtype=complex)
    
    # Store suffix products
    suffix_list = [(suffix_M00.copy(), suffix_M01.copy(), suffix_M10.copy(), suffix_M11.copy())]
    
    for i in range(n_layers - 1, -1, -1):
        B00, B01, B10, B11 = B_list[i][:4]
        
        new_sM00 = B00 * suffix_M00 + B01 * suffix_M10
        new_sM01 = B00 * suffix_M01 + B01 * suffix_M11
        new_sM10 = B10 * suffix_M00 + B11 * suffix_M10
        new_sM11 = B10 * suffix_M01 + B11 * suffix_M11
        
        suffix_M00, suffix_M01, suffix_M10, suffix_M11 = new_sM00, new_sM01, new_sM10, new_sM11
        suffix_list.insert(0, (suffix_M00.copy(), suffix_M01.copy(), suffix_M10.copy(), suffix_M11.copy()))
    
    # Now compute gradient for each layer
    grad = np.zeros(n_layers)
    
    for i in range(n_layers):
        B00, B01, B10, B11, dB00_dd, dB01_dd, dB10_dd, dB11_dd = B_list[i]
        
        # M_before_i
        Mb00, Mb01, Mb10, Mb11 = M_list[i]
        
        # M_after_i (suffix after layer i)
        # suffix_list[i] = B_i * B_{i+1} * ... * B_{n-1}
        # We need B_{i+1} * ... * B_{n-1}
        if i < n_layers - 1:
            Ma00, Ma01, Ma10, Ma11 = suffix_list[i + 1]
        else:
            Ma00 = np.ones(n_wl, dtype=complex)
            Ma01 = np.zeros(n_wl, dtype=complex)
            Ma10 = np.zeros(n_wl, dtype=complex)
            Ma11 = np.ones(n_wl, dtype=complex)
        
        # dM_total_dd_i = Mb * dB_dd * Ma
        # dB_dd matrix:
        # [[dB00_dd, dB01_dd],
        #  [dB10_dd, dB11_dd]]
        
        # (Mb * dB_dd)[0,0] = Mb00*dB00_dd + Mb01*dB10_dd
        # (Mb * dB_dd)[0,1] = Mb00*dB01_dd + Mb01*dB11_dd
        # (Mb * dB_dd)[1,0] = Mb10*dB00_dd + Mb11*dB10_dd
        # (Mb * dB_dd)[1,1] = Mb10*dB01_dd + Mb11*dB11_dd
        
        tmp00 = Mb00 * dB00_dd + Mb01 * dB10_dd
        tmp01 = Mb00 * dB01_dd + Mb01 * dB11_dd
        tmp10 = Mb10 * dB00_dd + Mb11 * dB10_dd
        tmp11 = Mb10 * dB01_dd + Mb11 * dB11_dd
        
        # (tmp * Ma)[0,0] = tmp00*Ma00 + tmp01*Ma10
        # (tmp * Ma)[0,1] = tmp00*Ma01 + tmp01*Ma11
        # (tmp * Ma)[1,0] = tmp10*Ma00 + tmp11*Ma10
        # (tmp * Ma)[1,1] = tmp10*Ma01 + tmp11*Ma11
        
        dM00 = tmp00 * Ma00 + tmp01 * Ma10
        dM01 = tmp00 * Ma01 + tmp01 * Ma11
        dM10 = tmp10 * Ma00 + tmp11 * Ma10
        dM11 = tmp10 * Ma01 + tmp11 * Ma11
        
        # dR/dd_i = Re(dR_dM00 * conj(dM00) + dR_dM01 * conj(dM01) + dR_dM10 * conj(dM10) + dR_dM11 * conj(dM11))
        # Actually for complex: dR/dd_i = Re(sum_jk dR_dM_jk * conj(dM_jk/dd_i))
        # But since R is real, dR/dd_i = 2*Re(sum_jk (dR/dM_jk)_real * ...) 
        # Simpler: dR/dd_i = 2 * Re( dR_dM00 * conj(dM00) + dR_dM01 * conj(dM01) + dR_dM10 * conj(dM10) + dR_dM11 * conj(dM11) )
        # Wait, R = |r|^2 is real. The derivative is:
        # dR/dd_i = 2*Re( (dr/dd_i) * conj(r) )
        # But we already have dR_dM which accounts for the full chain.
        # dR/dd_i = sum_jk (dR/dM_jk) * (dM_jk/dd_i) but with complex care.
        # Since R is real: dR/dd_i = Re( sum_jk dR_dM_jk * conj(dM_jk/dd_i) )
        # Actually the proper formula for real-valued R with complex intermediates:
        # dR/dd_i = 2*Re[ (dR/dM00)*conj(dM00/dd_i) + (dR/dM01)*conj(dM01/dd_i) + (dR/dM10)*conj(dM10/dd_i) + (dR/dM11)*conj(dM11/dd_i) ]
        # Hmm, let me think more carefully.
        # R = f(M) where f is real-valued. dR = sum_jk (∂R/∂Re(M_jk)) dRe(M_jk) + (∂R/∂Im(M_jk)) dIm(M_jk)
        # = Re[ sum_jk (∂R/∂M_jk) * conj(dM_jk) ] where ∂R/∂M_jk = ∂R/∂Re(M_jk) - i*∂R/∂Im(M_jk)
        # So dR/dd_i = Re[ sum_jk dR_dM_jk * conj(dM_jk/dd_i) ]
        
        val = (dR_dM00 * np.conj(dM00) + dR_dM01 * np.conj(dM01) + 
               dR_dM10 * np.conj(dM10) + dR_dM11 * np.conj(dM11))
        grad[i] = np.mean(np.real(val)) * 100.0  # *100 for percent
    
    return obj, grad

def _objective(thicknesses, indices, wavelengths_nm, ambient_idx, substrate_idx,
               max_layer_nm, max_total_nm):
    """
    Objective: mean reflectance in percent. Returns large penalty if constraints violated.
    """
    thicknesses = np.asarray(thicknesses, dtype=float)
    if np.any(thicknesses < -1e-6) or np.any(thicknesses > max_layer_nm + 1e-6):
        return 100.0
    if np.sum(thicknesses) > max_total_nm + 1e-6:
        return 100.0
    
    thicknesses_clipped = np.clip(thicknesses, 0.0, max_layer_nm)
    if np.sum(thicknesses_clipped) > max_total_nm:
        return 100.0
    
    R = _compute_reflectance(wavelengths_nm, indices, thicknesses_clipped,
                             ambient_idx, substrate_idx)
    return np.mean(R) * 100.0

def _nelder_mead(f, x0, bounds, max_iter=2000, tol=1e-10, time_deadline=None):
    """
    Nelder-Mead simplex with bounds.
    """
    n = len(x0)
    lo_bounds = np.array([b[0] for b in bounds])
    hi_bounds = np.array([b[1] for b in bounds])
    x0 = np.clip(np.asarray(x0, dtype=float), lo_bounds, hi_bounds)
    
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    
    for i in range(n):
        simplex[i + 1] = x0.copy()
        lo, hi = bounds[i]
        step = 0.05 * (hi - lo) if hi > lo else 0.05
        simplex[i + 1][i] += step
    
    fvals = np.array([f(simplex[i]) for i in range(n + 1)])
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for it in range(max_iter):
        if time_deadline is not None and time.time() > time_deadline:
            break
        
        idx = np.argsort(fvals)
        simplex = simplex[idx]
        fvals = fvals[idx]
        
        if np.max(fvals) - np.min(fvals) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        x_r = centroid + alpha * (centroid - simplex[-1])
        x_r = np.clip(x_r, lo_bounds, hi_bounds)
        f_r = f(x_r)
        
        if f_r < fvals[0]:
            x_e = centroid + gamma * (x_r - centroid)
            x_e = np.clip(x_e, lo_bounds, hi_bounds)
            f_e = f(x_e)
            
            if f_e < f_r:
                simplex[-1] = x_e
                fvals[-1] = f_e
            else:
                simplex[-1] = x_r
                fvals[-1] = f_r
        elif f_r < fvals[-2]:
            simplex[-1] = x_r
            fvals[-1] = f_r
        else:
            x_c = centroid + rho * (simplex[-1] - centroid)
            x_c = np.clip(x_c, lo_bounds, hi_bounds)
            f_c = f(x_c)
            
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], lo_bounds, hi_bounds)
                    fvals[i] = f(simplex[i])
    
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]

def _generate_initial_designs(indices, n_layers, center_wl, max_total_nm, max_layer_nm, n_starts=20):
    """
    Generate diverse, physically-motivated initial designs.
    """
    designs = []
    
    # Quarter-wave stack
    qw = np.array([center_wl / (4.0 * n_i) for n_i in indices])
    if np.sum(qw) <= max_total_nm:
        designs.append(qw.copy())
    else:
        designs.append(qw * (max_total_nm / np.sum(qw)))
    
    # Half-wave stack
    hw = np.array([center_wl / (2.0 * n_i) for n_i in indices])
    if np.sum(hw) <= max_total_nm:
        designs.append(hw.copy())
    else:
        designs.append(hw * (max_total_nm / np.sum(hw)))
    
    # Various fractions of quarter-wave
    for frac in [0.5, 0.75, 1.25, 1.5]:
        trial = np.array([frac * center_wl / (4.0 * n_i) for n_i in indices])
        if np.sum(trial) <= max_total_nm:
            designs.append(trial)
        else:
            designs.append(trial * (max_total_nm / np.sum(trial)))
    
    # Random designs with physical constraints
    rng = np.random.RandomState(42)
    for _ in range(max(0, n_starts - len(designs))):
        trial = rng.uniform(0.1, 1.0, size=n_layers)
        trial = trial * (center_wl / 4.0) / np.array(indices)
        if np.sum(trial) > max_total_nm:
            trial = trial * (max_total_nm / np.sum(trial))
        trial = np.clip(trial, 0, max_layer_nm)
        designs.append(trial)
    
    # Ensure all are valid
    valid_designs = []
    for d in designs:
        d = np.clip(d, 0, max_layer_nm)
        if np.sum(d) <= max_total_nm:
            valid_designs.append(d)
    
    if not valid_designs:
        valid_designs.append(np.array([center_wl / (4.0 * indices[0])]))
    
    return valid_designs

def _generate_layer_configs(n_layers, max_layers):
    """
    Generate diverse layer label configurations (not just alternating).
    """
    configs = []
    if n_layers > max_layers:
        return configs
    
    # Alternating starting with L
    configs.append(['L' if i % 2 == 0 else 'H' for i in range(n_layers)])
    # Alternating starting with H
    configs.append(['H' if i % 2 == 0 else 'L' for i in range(n_layers)])
    
    # Some non-alternating patterns for broader search
    if n_layers >= 4:
        # LLHH pattern
        configs.append(['L' if (i // 2) % 2 == 0 else 'H' for i in range(n_layers)])
        # HHLL pattern
        configs.append(['H' if (i // 2) % 2 == 0 else 'L' for i in range(n_layers)])
    
    if n_layers >= 6:
        # LLLHHH
        configs.append(['L' if (i // 3) % 2 == 0 else 'H' for i in range(n_layers)])
        # HHHLLL
        configs.append(['H' if (i // 3) % 2 == 0 else 'L' for i in range(n_layers)])
    
    # Remove duplicates
    seen = set()
    unique_configs = []
    for c in configs:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_configs.append(c)
    
    return unique_configs

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
    
    # Try multiple layer counts and configurations
    layer_counts = [5, 7, 9, 11, 13]
    layer_counts = [lc for lc in layer_counts if lc <= max_layers]
    
    for n_layers in layer_counts:
        if time.time() > deadline - 15:
            break
        
        configs = _generate_layer_configs(n_layers, max_layers)
        
        for labels in configs:
            if time.time() > deadline - 10:
                break
            
            indices = [materials[label] for label in labels]
            n = len(indices)
            bounds = [(0.0, float(max_layer_nm))] * n
            
            # Create objective and gradient functions for L-BFGS-B
            def obj_func(x, indices=indices):
                return _objective(x, indices, wavelengths_nm, ambient_idx, substrate_idx,
                                  max_layer_nm, max_total_nm)
            
            def obj_grad_func(x, indices=indices):
                val, grad = _objective_and_gradient(x, indices, wavelengths_nm, ambient_idx,
                                                    substrate_idx, max_layer_nm, max_total_nm)
                return val, grad
            
            # Generate initial designs
            initial_designs = _generate_initial_designs(indices, n, center_wl,
                                                        max_total_nm, max_layer_nm, n_starts=12)
            
            best_x_local = None
            best_f_local = np.inf
            
            for x0 in initial_designs:
                if time.time() > deadline - 8:
                    break
                
                x0 = np.clip(x0, 0, max_layer_nm)
                if np.sum(x0) > max_total_nm:
                    x0 = x0 * (max_total_nm / np.sum(x0))
                x0 = np.clip(x0, 0, max_layer_nm)
                
                # Use L-BFGS-B with analytic gradient
                lbfgs_deadline = min(time.time() + 8.0, deadline - 5)
                try:
                    result = minimize(
                        obj_func,
                        x0,
                        jac=obj_grad_func,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10,
                                  'maxfun': 2000}
                    )
                    if result.fun < best_f_local:
                        best_f_local = result.fun
                        best_x_local = result.x.copy()
                except Exception:
                    pass
                
                # If L-BFGS-B didn't converge well, try NM as fallback
                if time.time() < lbfgs_deadline - 2:
                    nm_deadline = min(time.time() + 4.0, deadline - 5)
                    try:
                        x_nm, f_nm = _nelder_mead(obj_func, x0, bounds, max_iter=500,
                                                   tol=1e-10, time_deadline=nm_deadline)
                        if f_nm < best_f_local:
                            best_f_local = f_nm
                            best_x_local = x_nm.copy()
                    except Exception:
                        pass
            
            # Track global best
            if best_x_local is not None and best_f_local < best_score:
                best_score = best_f_local
                best_x_local_clipped = np.clip(best_x_local, 0, max_layer_nm)
                if np.sum(best_x_local_clipped) > max_total_nm:
                    best_x_local_clipped = best_x_local_clipped * (max_total_nm / np.sum(best_x_local_clipped))
                best_thicknesses = best_x_local_clipped.copy()
                best_labels = labels.copy()
    
    # If no valid result found, use a simple quarter-wave AR coating
    if best_thicknesses is None:
        best_labels = ['L']
        best_thicknesses = np.array([center_wl / (4.0 * materials["L"])])
    
    # Final constraint check
    thicknesses = np.array(best_thicknesses)
    thicknesses = np.clip(thicknesses, 0, max_layer_nm)
    if np.sum(thicknesses) > max_total_nm:
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
    
    return best_labels, thicknesses.tolist()
