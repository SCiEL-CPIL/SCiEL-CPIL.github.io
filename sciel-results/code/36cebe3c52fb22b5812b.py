"""
Broadband antireflection coating design via multi-start Nelder-Mead + gradient polish + coordinate descent.

Strategy:
- Generate diverse, physically-motivated initial designs
- Run Nelder-Mead from each start (smooth objective converges fast)
- Polish best results with gradient descent
- Add coordinate descent for final refinement
- Select the best result within the time budget
"""
import numpy as np
import time

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

def _compute_reflectance_and_grad(wavelengths_nm, indices, thicknesses_nm, ambient_idx, substrate_idx):
    """
    Compute mean reflectance (percent) and its analytic gradient w.r.t. each thickness.
    Uses reverse-mode chain rule through the characteristic matrix.
    """
    n_layers = len(indices)
    if n_layers == 0:
        r0 = (ambient_idx - substrate_idx) / (ambient_idx + substrate_idx)
        return r0 * r0 * 100.0, np.zeros(0)
    
    k = 2.0 * np.pi / wavelengths_nm
    n_w = len(wavelengths_nm)
    
    # Forward pass: store per-layer B matrices and accumulated M
    # M is the product of B matrices from layer 0 to layer i
    M = np.zeros((n_layers + 1, n_w, 2, 2), dtype=complex)
    M[0, :, 0, 0] = 1.0
    M[0, :, 1, 1] = 1.0
    
    B = np.zeros((n_layers, n_w, 2, 2), dtype=complex)
    for i in range(n_layers):
        n_i = indices[i]
        d_i = thicknesses_nm[i]
        delta = k * n_i * d_i
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        B[i, :, 0, 0] = cos_d
        B[i, :, 0, 1] = 1j * sin_d / n_i
        B[i, :, 1, 0] = 1j * n_i * sin_d
        B[i, :, 1, 1] = cos_d
        # M[i+1] = M[i] @ B[i]
        M[i+1, :, 0, 0] = M[i, :, 0, 0] * B[i, :, 0, 0] + M[i, :, 0, 1] * B[i, :, 1, 0]
        M[i+1, :, 0, 1] = M[i, :, 0, 0] * B[i, :, 0, 1] + M[i, :, 0, 1] * B[i, :, 1, 1]
        M[i+1, :, 1, 0] = M[i, :, 1, 0] * B[i, :, 0, 0] + M[i, :, 1, 1] * B[i, :, 1, 0]
        M[i+1, :, 1, 1] = M[i, :, 1, 0] * B[i, :, 0, 1] + M[i, :, 1, 1] * B[i, :, 1, 1]
    
    M_final = M[n_layers]
    M00 = M_final[:, 0, 0]
    M01 = M_final[:, 0, 1]
    M10 = M_final[:, 1, 0]
    M11 = M_final[:, 1, 1]
    
    Y_out = substrate_idx
    Y_in = (M10 + M11 * Y_out) / (M00 + M01 * Y_out)
    r = (ambient_idx - Y_in) / (ambient_idx + Y_in)
    R = np.abs(r) ** 2
    mean_R = np.mean(R) * 100.0
    
    # Backward pass: compute dR/dM elements
    # Y_in = (M10 + M11*Ys) / (M00 + M01*Ys)
    # r = (na - Y_in) / (na + Y_in)
    # R = |r|^2
    # dR/dY_in = 2*Re(r * conj(dr/dY_in)) where dr/dY_in = -2*na/(na+Y_in)^2
    # Actually: r = (na - Y)/(na + Y), dr/dY = -2*na/(na+Y)^2
    # R = r * conj(r), dR/dY = conj(r)*dr/dY + r*d(conj(r))/d(conj(Y))
    # For complex Y: dR/dY = conj(r) * dr/dY + r * d(conj(r))/d(conj(Y))
    # Since R is real: dR/dY = 2*Re(conj(r) * dr/dY)
    # But we need dR/dM10, dR/dM11, dR/dM00, dR/dM01
    
    # Y_in = N / D where N = M10 + M11*Ys, D = M00 + M01*Ys
    # dY_in/dM10 = 1/D, dY_in/dM11 = Ys/D, dY_in/dM00 = -N/D^2, dY_in/dM01 = -Ys*N/D^2
    # r = (na - Y)/(na + Y), dr/dY = -2*na/(na+Y)^2
    # R = |r|^2, dR/dY = 2*Re(conj(r) * dr/dY)
    
    dr_dY = -2.0 * ambient_idx / (ambient_idx + Y_in) ** 2
    dR_dY = 2.0 * np.real(np.conj(r) * dr_dY)
    
    N = M10 + M11 * Y_out
    D = M00 + M01 * Y_out
    
    dR_dM10 = dR_dY / D
    dR_dM11 = dR_dY * Y_out / D
    dR_dM00 = -dR_dY * N / D**2
    dR_dM01 = -dR_dY * Y_out * N / D**2
    
    # Now backpropagate through M = B[0] @ B[1] @ ... @ B[n-1]
    # M[i] = M[i-1] @ B[i-1]
    # dR/dB[i-1] = M[i-1]^T (in matrix sense) ... 
    # Actually for matrices: M = A @ B, dR/dA = (dR/dM) @ B^T, dR/dB = A^T @ (dR/dM)
    # But here it's element-wise per wavelength, so treat each wavelength independently.
    
    # dR/dM_final = [[dR_dM00, dR_dM01], [dR_dM10, dR_dM11]]
    dM = np.zeros((n_w, 2, 2), dtype=complex)
    dM[:, 0, 0] = dR_dM00
    dM[:, 0, 1] = dR_dM01
    dM[:, 1, 0] = dR_dM10
    dM[:, 1, 1] = dR_dM11
    
    # Backward: dR/dB[i] from dR/dM[i+1]
    # M[i+1] = M[i] @ B[i]
    # dR/dM[i] = dR/dM[i+1] @ B[i]^T  (transpose, not conjugate transpose, since we're doing real-valued backprop)
    # Actually for complex: we need to be careful. Let's use the fact that R is real.
    # dR/dM[i] = B[i]^T @ dR/dM[i+1] ... no.
    # If M_new = M_old @ B, then dR/dM_old = dR/dM_new @ B^T (for real matrices)
    # For complex, since R is real, dR/dM_old = (dR/dM_new @ B^T) but we need to handle conjugates.
    # Simpler: use the fact that dR = trace(dM^T @ dR/dM) for real, or for complex use the Wirtinger calculus.
    # Let me just do: dR/dB[i] = M[i]^T @ dR/dM[i+1] (element-wise per wavelength, treating as 2x2 real+imag)
    
    # Actually, let me just do it properly with the chain rule for 2x2 matrix product.
    # M_new = M_old @ B
    # M_new[0,0] = M_old[0,0]*B[0,0] + M_old[0,1]*B[1,0]
    # M_new[0,1] = M_old[0,0]*B[0,1] + M_old[0,1]*B[1,1]
    # M_new[1,0] = M_old[1,0]*B[0,0] + M_old[1,1]*B[1,0]
    # M_new[1,1] = M_old[1,0]*B[0,1] + M_old[1,1]*B[1,1]
    
    # dR/dB[0,0] = dR/dM_new[0,0] * M_old[0,0] + dR/dM_new[0,1] * M_old[0,0] ... no
    # dR/dB[0,0] = dR/dM_new[0,0] * M_old[0,0] + dR/dM_new[1,0] * M_old[1,0]
    # Wait, M_new[0,0] depends on B[0,0] with coefficient M_old[0,0]
    # M_new[1,0] depends on B[0,0] with coefficient M_old[1,0]
    # So dR/dB[0,0] = M_old[0,0] * dR/dM_new[0,0] + M_old[1,0] * dR/dM_new[1,0]
    
    # Similarly:
    # dR/dB[0,1] = M_old[0,0] * dR/dM_new[0,1] + M_old[1,0] * dR/dM_new[1,1]
    # dR/dB[1,0] = M_old[0,1] * dR/dM_new[0,0] + M_old[1,1] * dR/dM_new[1,0]
    # dR/dB[1,1] = M_old[0,1] * dR/dM_new[0,1] + M_old[1,1] * dR/dM_new[1,1]
    
    # And dR/dM_old:
    # dR/dM_old[0,0] = B[0,0]*dR/dM_new[0,0] + B[0,1]*dR/dM_new[0,1]
    # dR/dM_old[0,1] = B[1,0]*dR/dM_new[0,0] + B[1,1]*dR/dM_new[0,1]
    # dR/dM_old[1,0] = B[0,0]*dR/dM_new[1,0] + B[0,1]*dR/dM_new[1,1]
    # dR/dM_old[1,1] = B[1,0]*dR/dM_new[1,0] + B[1,1]*dR/dM_new[1,1]
    
    # Since R is real, we take the real part at the end.
    
    grad = np.zeros(n_layers)
    for i in range(n_layers - 1, -1, -1):
        # dR/dB[i] from dM (which is dR/dM[i+1])
        dR_dB00 = M[i, :, 0, 0] * dM[:, 0, 0] + M[i, :, 1, 0] * dM[:, 1, 0]
        dR_dB01 = M[i, :, 0, 0] * dM[:, 0, 1] + M[i, :, 1, 0] * dM[:, 1, 1]
        dR_dB10 = M[i, :, 0, 1] * dM[:, 0, 0] + M[i, :, 1, 1] * dM[:, 1, 0]
        dR_dB11 = M[i, :, 0, 1] * dM[:, 0, 1] + M[i, :, 1, 1] * dM[:, 1, 1]
        
        # B[i] = [[cos(delta), 1j*sin(delta)/n], [1j*n*sin(delta), cos(delta)]]
        # delta = k * n_i * d_i
        # dB00/ddelta = -sin(delta), dB01/ddelta = 1j*cos(delta)/n, dB10/ddelta = 1j*n*cos(delta), dB11/ddelta = -sin(delta)
        # ddelta/dd_i = k * n_i
        
        n_i = indices[i]
        d_i = thicknesses_nm[i]
        delta = k * n_i * d_i
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        dB00_dd = -sin_d
        dB01_dd = 1j * cos_d / n_i
        dB10_dd = 1j * n_i * cos_d
        dB11_dd = -sin_d
        
        dd_dd_i = k * n_i
        
        dR_dd_i = np.real(
            dR_dB00 * dB00_dd + dR_dB01 * dB01_dd + dR_dB10 * dB10_dd + dR_dB11 * dB11_dd
        ) * dd_dd_i
        
        grad[i] = np.mean(dR_dd_i)
        
        # Update dM = dR/dM[i]
        new_dM00 = B[i, :, 0, 0] * dM[:, 0, 0] + B[i, :, 0, 1] * dM[:, 0, 1]
        new_dM01 = B[i, :, 1, 0] * dM[:, 0, 0] + B[i, :, 1, 1] * dM[:, 0, 1]
        new_dM10 = B[i, :, 0, 0] * dM[:, 1, 0] + B[i, :, 0, 1] * dM[:, 1, 1]
        new_dM11 = B[i, :, 1, 0] * dM[:, 1, 0] + B[i, :, 1, 1] * dM[:, 1, 1]
        dM[:, 0, 0] = new_dM00
        dM[:, 0, 1] = new_dM01
        dM[:, 1, 0] = new_dM10
        dM[:, 1, 1] = new_dM11
    
    return mean_R, grad

def _polish_with_gradient_descent(f, x0, bounds, max_iter=500, time_deadline=None, 
                                    indices=None, wavelengths_nm=None, ambient_idx=None, 
                                    substrate_idx=None):
    """
    L-BFGS-B polish using analytic gradient when available, else numerical.
    """
    from scipy.optimize import minimize
    
    n = len(x0)
    lo_bounds = np.array([b[0] for b in bounds])
    hi_bounds = np.array([b[1] for b in bounds])
    x = np.clip(np.asarray(x0, dtype=float), lo_bounds, hi_bounds)
    
    # Handle total thickness constraint by projecting
    max_total = float(np.sum(hi_bounds))  # fallback
    if 'max_total_nm' in dir():
        max_total = max_total_nm
    
    def objective_and_grad(x):
        x = np.clip(x, lo_bounds, hi_bounds)
        if np.sum(x) > max_total:
            x = x * (max_total / np.sum(x))
            x = np.clip(x, lo_bounds, hi_bounds)
        fval = f(x)
        if indices is not None and wavelengths_nm is not None and ambient_idx is not None and substrate_idx is not None:
            _, grad = _compute_reflectance_and_grad(wavelengths_nm, indices, x, ambient_idx, substrate_idx)
        else:
            eps = 1e-6
            grad = np.zeros(n)
            for i in range(n):
                xp = x.copy()
                xm = x.copy()
                xp[i] = min(x[i] + eps, hi_bounds[i])
                xm[i] = max(x[i] - eps, lo_bounds[i])
                grad[i] = (f(xp) - f(xm)) / (xp[i] - xm[i])
        return fval, grad
    
    res = minimize(objective_and_grad, x, method='L-BFGS-B', 
                   bounds=list(zip(lo_bounds, hi_bounds)),
                   options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-6,
                            'maxfun': max_iter * n * 2})
    
    x_opt = np.clip(res.x, lo_bounds, hi_bounds)
    if np.sum(x_opt) > max_total:
        x_opt = x_opt * (max_total / np.sum(x_opt))
        x_opt = np.clip(x_opt, lo_bounds, hi_bounds)
    f_opt = f(x_opt)
    
    return x_opt, f_opt

def _coordinate_descent(f, x0, bounds, max_iter=100, time_deadline=None, max_total_nm=1000.0):
    """
    Coordinate descent for final refinement.
    """
    n = len(x0)
    lo_bounds = np.array([b[0] for b in bounds])
    hi_bounds = np.array([b[1] for b in bounds])
    x = np.clip(np.asarray(x0, dtype=float), lo_bounds, hi_bounds)
    fval = f(x)
    
    for it in range(max_iter):
        if time_deadline is not None and time.time() > time_deadline:
            break
        
        improved = False
        for i in range(n):
            # Try to optimize one coordinate at a time using golden section
            best_xi = x[i]
            best_fi = fval
            
            # Fine-grained local search
            for delta in np.linspace(-5, 5, 21):
                xi_new = x[i] + delta
                if xi_new < lo_bounds[i] or xi_new > hi_bounds[i]:
                    continue
                x_trial = x.copy()
                x_trial[i] = xi_new
                if np.sum(x_trial) > max_total_nm:
                    continue
                f_trial = f(x_trial)
                if f_trial < best_fi:
                    best_fi = f_trial
                    best_xi = xi_new
            
            if best_xi != x[i]:
                x[i] = best_xi
                fval = best_fi
                improved = True
        
        if not improved:
            break
    
    return x, fval

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
    for _ in range(n_starts - len(designs)):
        # Random thicknesses, then scale to fit budget
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
        # Fallback: single layer
        valid_designs.append(np.array([center_wl / (4.0 * indices[0])]))
    
    return valid_designs

def _design_direct(problem):
    """
    Direct physical construction of broadband AR coating.
    """
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
    
    for n_layers in [7, 9, 11, 13]:
        if n_layers > max_layers:
            continue
        
        labels_1 = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        labels_2 = ['H' if i % 2 == 0 else 'L' for i in range(n_layers)]
        
        for labels in [labels_1, labels_2]:
            indices = [materials[label] for label in labels]
            
            x0 = np.array([center_wl / (4.0 * n_i) for n_i in indices])
            
            if np.sum(x0) > max_total_nm:
                x0 = x0 * (max_total_nm / np.sum(x0))
            x0 = np.clip(x0, 0, max_layer_nm)
            
            f0 = _objective(x0, indices, wavelengths_nm, ambient_idx, substrate_idx,
                           max_layer_nm, max_total_nm)
            
            if f0 < best_score:
                best_score = f0
                best_labels = labels.copy()
                best_thicknesses = x0.copy()
    
    if best_labels is None:
        return ["L"], [center_wl / (4.0 * materials["L"])]
    
    return best_labels, best_thicknesses

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
    
    # Get initial design
    init_labels, init_thicknesses = _design_direct(problem)
    indices = [materials[label] for label in init_labels]
    n_layers = len(indices)
    bounds = [(0.0, float(max_layer_nm))] * n_layers
    
    def make_f(indices):
        def f(x):
            return _objective(x, indices, wavelengths_nm, ambient_idx, substrate_idx,
                              max_layer_nm, max_total_nm)
        return f
    
    f = make_f(indices)
    
    # Generate diverse initial designs
    initial_designs = _generate_initial_designs(indices, n_layers, center_wl, 
                                                 max_total_nm, max_layer_nm, n_starts=25)
    
    best_x = None
    best_f = np.inf
    
    # Multi-start Nelder-Mead
    for i, x0 in enumerate(initial_designs):
        if time.time() > deadline - 20:
            break
        
        x0 = np.clip(x0, 0, max_layer_nm)
        if np.sum(x0) > max_total_nm:
            x0 = x0 * (max_total_nm / np.sum(x0))
        
        # Run NM from this start
        nm_deadline = min(time.time() + 5.0, deadline - 15)
        try:
            x_nm, f_nm = _nelder_mead(f, x0, bounds, max_iter=1000, tol=1e-10, 
                                       time_deadline=nm_deadline)
            if f_nm < best_f:
                best_f = f_nm
                best_x = x_nm.copy()
        except Exception:
            pass
    
    # Polish best result with gradient descent (analytic gradient)
    if best_x is not None:
        gd_deadline = min(time.time() + 10.0, deadline - 5)
        try:
            x_gd, f_gd = _polish_with_gradient_descent(f, best_x, bounds, max_iter=200, 
                                                       time_deadline=gd_deadline,
                                                       indices=indices,
                                                       wavelengths_nm=wavelengths_nm,
                                                       ambient_idx=ambient_idx,
                                                       substrate_idx=substrate_idx)
            if f_gd < best_f:
                best_f = f_gd
                best_x = x_gd.copy()
        except Exception:
            pass
    
    # Add coordinate descent for final refinement
    if best_x is not None and time.time() < deadline - 10:
        cd_deadline = min(time.time() + 5.0, deadline - 3)
        try:
            x_cd, f_cd = _coordinate_descent(f, best_x, bounds, max_iter=50, 
                                             time_deadline=cd_deadline, max_total_nm=max_total_nm)
            if f_cd < best_f:
                best_f = f_cd
                best_x = x_cd.copy()
        except Exception:
            pass
    
    # Try a few more layer configurations if time permits
    if time.time() < deadline - 30:
        best_overall_x = best_x.copy() if best_x is not None else np.array(init_thicknesses)
        best_overall_f = best_f if best_x is not None else _objective(init_thicknesses, indices, 
                                                                      wavelengths_nm, ambient_idx, 
                                                                      substrate_idx, max_layer_nm, 
                                                                      max_total_nm)
        best_overall_labels = init_labels.copy()
        
        for n_try in [9, 11, 13]:
            if time.time() > deadline - 25:
                break
            if n_try > max_layers:
                continue
            
            for start_char in ['L', 'H']:
                if time.time() > deadline - 20:
                    break
                
                labels_try = [start_char if i % 2 == 0 else ('H' if start_char == 'L' else 'L') 
                              for i in range(n_try)]
                indices_try = [materials[label] for label in labels_try]
                n_try_layers = len(indices_try)
                bounds_try = [(0.0, float(max_layer_nm))] * n_try_layers
                
                f_try = make_f(indices_try)
                
                # Generate initial designs for this configuration
                initial_designs_try = _generate_initial_designs(indices_try, n_try_layers, 
                                                                 center_wl, max_total_nm, 
                                                                 max_layer_nm, n_starts=15)
                
                best_x_try = None
                best_f_try = np.inf
                
                for x0 in initial_designs_try:
                    if time.time() > deadline - 15:
                        break
                    
                    x0 = np.clip(x0, 0, max_layer_nm)
                    if np.sum(x0) > max_total_nm:
                        x0 = x0 * (max_total_nm / np.sum(x0))
                    
                    nm_deadline_try = min(time.time() + 3.0, deadline - 10)
                    try:
                        x_nm_try, f_nm_try = _nelder_mead(f_try, x0, bounds_try, max_iter=800, 
                                                           tol=1e-10, time_deadline=nm_deadline_try)
                        if f_nm_try < best_f_try:
                            best_f_try = f_nm_try
                            best_x_try = x_nm_try.copy()
                    except Exception:
                        pass
                
                # Polish with gradient descent (analytic gradient)
                if best_x_try is not None:
                    gd_deadline_try = min(time.time() + 5.0, deadline - 5)
                    try:
                        x_gd_try, f_gd_try = _polish_with_gradient_descent(f_try, best_x_try, bounds_try, 
                                                                           max_iter=100, time_deadline=gd_deadline_try,
                                                                           indices=indices_try,
                                                                           wavelengths_nm=wavelengths_nm,
                                                                           ambient_idx=ambient_idx,
                                                                           substrate_idx=substrate_idx)
                        if f_gd_try < best_f_try:
                            best_f_try = f_gd_try
                            best_x_try = x_gd_try.copy()
                    except Exception:
                        pass
                
                # Add coordinate descent for this configuration
                if best_x_try is not None and time.time() < deadline - 8:
                    cd_deadline_try = min(time.time() + 3.0, deadline - 4)
                    try:
                        x_cd_try, f_cd_try = _coordinate_descent(f_try, best_x_try, bounds_try, 
                                                                 max_iter=30, time_deadline=cd_deadline_try,
                                                                 max_total_nm=max_total_nm)
                        if f_cd_try < best_f_try:
                            best_f_try = f_cd_try
                            best_x_try = x_cd_try.copy()
                    except Exception:
                        pass
                
                if best_x_try is not None and best_f_try < best_overall_f:
                    best_overall_x = best_x_try.copy()
                    best_overall_f = best_f_try
                    best_overall_labels = labels_try.copy()
        
        if best_overall_f < best_f:
            best_x = best_overall_x
            best_f = best_overall_f
            init_labels = best_overall_labels
    
    # Final constraint check
    if best_x is None:
        best_x = np.array(init_thicknesses)
    
    thicknesses = np.array(best_x)
    thicknesses = np.clip(thicknesses, 0, max_layer_nm)
    if np.sum(thicknesses) > max_total_nm:
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
    
    return init_labels, thicknesses.tolist()
