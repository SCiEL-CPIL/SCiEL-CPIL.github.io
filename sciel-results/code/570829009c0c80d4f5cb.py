"""
Broadband antireflection coating design via multi-start Nelder-Mead + gradient polish.

Strategy:
- Generate diverse, physically-motivated initial designs
- Run Nelder-Mead from each start (smooth objective converges fast)
- Polish best results with gradient descent
- Select the best result within the time budget
"""
import numpy as np
import time
import math

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
    if np.any(thicknesses < -1e-9) or np.any(thicknesses > max_layer_nm + 1e-9):
        return 100.0
    if np.sum(thicknesses) > max_total_nm + 1e-9:
        return 100.0
    
    thicknesses_clipped = np.clip(thicknesses, 0.0, max_layer_nm)
    if np.sum(thicknesses_clipped) > max_total_nm:
        return 100.0
    
    R = _compute_reflectance(wavelengths_nm, indices, thicknesses_clipped, 
                             ambient_idx, substrate_idx)
    return np.mean(R) * 100.0

def _nelder_mead(f, x0, bounds, max_iter=3000, tol=1e-12, time_deadline=None):
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

def _gradient_descent_polish(f, x0, bounds, max_iter=1000, tol=1e-10, time_deadline=None):
    """
    Simple gradient descent with numerical gradients and adaptive step size.
    """
    n = len(x0)
    lo_bounds = np.array([b[0] for b in bounds])
    hi_bounds = np.array([b[1] for b in bounds])
    x = np.clip(np.asarray(x0, dtype=float), lo_bounds, hi_bounds)
    f_val = f(x)
    
    step_size = 0.1 * (hi_bounds - lo_bounds)
    
    for it in range(max_iter):
        if time_deadline is not None and time.time() > time_deadline:
            break
        
        # Compute numerical gradient
        grad = np.zeros(n)
        for i in range(n):
            h = 1e-6 * max(1.0, abs(x[i]))
            x_plus = x.copy()
            x_plus[i] += h
            x_plus = np.clip(x_plus, lo_bounds, hi_bounds)
            
            x_minus = x.copy()
            x_minus[i] -= h
            x_minus = np.clip(x_minus, lo_bounds, hi_bounds)
            
            f_plus = f(x_plus)
            f_minus = f(x_minus)
            
            if x[i] == lo_bounds[i]:
                grad[i] = (f_plus - f_val) / h
            elif x[i] == hi_bounds[i]:
                grad[i] = (f_val - f_minus) / h
            else:
                grad[i] = (f_plus - f_minus) / (2 * h)
        
        # Check convergence
        if np.max(np.abs(grad)) < tol:
            break
        
        # Update
        x_new = x - step_size * grad
        x_new = np.clip(x_new, lo_bounds, hi_bounds)
        
        f_new = f(x_new)
        if f_new < f_val:
            x = x_new
            f_val = f_new
            step_size *= 1.05 # Increase step slightly if progress
        else:
            step_size *= 0.5 # Decrease step if no progress
            if step_size < 1e-8:
                break
    
    return x, f_val

def _polish(f, x0, bounds, time_deadline=None):
    """
    Polish with Nelder-Mead.
    """
    return _nelder_mead(f, x0, bounds, max_iter=2000, tol=1e-12, time_deadline=time_deadline)

def _design_for_layer_count(n_layers, materials, ambient_idx, substrate_idx, 
                            wavelengths_nm, center_wl, max_total_nm, max_layer_nm,
                            start_char, time_deadline):
    """
    Design for a specific layer count and starting material using multi-start NM + GD.
    """
    labels = [start_char if i % 2 == 0 else ('H' if start_char == 'L' else 'L') 
              for i in range(n_layers)]
    indices = [materials[label] for label in labels]
    
    bounds = [(0.0, float(max_layer_nm))] * n_layers
    
    def make_f(indices):
        def f(x):
            return _objective(x, indices, wavelengths_nm, ambient_idx, substrate_idx,
                              max_layer_nm, max_total_nm)
        return f
    
    f = make_f(indices)
    
    # Generate diverse initial guesses
    rng = np.random.RandomState(42 + n_layers + (1 if start_char == 'H' else 0))
    initial_guesses = []
    
    # 1. Quarter-wave stack
    x_qw = np.array([center_wl / (4.0 * n_i) for n_i in indices])
    if np.sum(x_qw) > max_total_nm:
        x_qw = x_qw * (max_total_nm / np.sum(x_qw))
    x_qw = np.clip(x_qw, 0, max_layer_nm)
    initial_guesses.append(x_qw)
    
    # 2. Random guesses
    for _ in range(10):
        x_rand = rng.uniform(0, max_layer_nm, size=n_layers)
        if np.sum(x_rand) > max_total_nm:
            x_rand = x_rand * (max_total_nm / np.sum(x_rand))
        initial_guesses.append(x_rand)
        
    # 3. Physically motivated: alternate thicknesses
    for factor in [0.5, 1.5, 2.0]:
        x_phys = np.array([center_wl * factor / (4.0 * n_i) for n_i in indices])
        if np.sum(x_phys) > max_total_nm:
            x_phys = x_phys * (max_total_nm / np.sum(x_phys))
        x_phys = np.clip(x_phys, 0, max_layer_nm)
        initial_guesses.append(x_phys)

    best_x = None
    best_f = np.inf
    
    # Run Nelder-Mead from each start, then GD polish
    for x0 in initial_guesses:
        if time.time() > time_deadline - 5:
            break
            
        try:
            # Nelder-Mead
            x_nm, f_nm = _nelder_mead(f, x0, bounds, max_iter=500, tol=1e-10, time_deadline=time_deadline - 5)
            
            # Gradient Descent Polish
            x_gd, f_gd = _gradient_descent_polish(f, x_nm, bounds, max_iter=200, time_deadline=time_deadline - 2)
            
            if f_gd < best_f:
                best_f = f_gd
                best_x = x_gd
        except Exception:
            continue
            
    if best_x is None:
        best_x = initial_guesses[0]
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
    
    # Try different layer counts and starting materials
    # Focus on odd layer counts which often perform better for AR coatings
    layer_configs = []
    for n_layers in [3, 5, 7, 9, 11, 13]:
        if n_layers > max_layers:
            continue
        for start_char in ['L', 'H']:
            layer_configs.append((n_layers, start_char))
    
    # Sort by expected performance or just iterate
    for n_layers, start_char in layer_configs:
        if time.time() > deadline - 15:
            break
        
        try:
            labels, thicknesses, score = _design_for_layer_count(
                n_layers, materials, ambient_idx, substrate_idx,
                wavelengths_nm, center_wl, max_total_nm, max_layer_nm,
                start_char, deadline - 5
            )
            
            if score < best_score:
                best_score = score
                best_labels = labels
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
