"""Broadband antireflection coating design via multi-start local optimization with known good structures."""
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

def _objective(params, problem, wavelengths_nm):
    """
    Objective function: mean reflectance in percent.
    params: flat array of thicknesses
    """
    max_layer_nm = problem.get("max_layer_nm", 400)
    max_total_nm = problem.get("max_total_nm", 1000)
    
    thicknesses = np.clip(params, 0, max_layer_nm)
    
    if np.sum(thicknesses) > max_total_nm:
        return 100.0 + (np.sum(thicknesses) - max_total_nm)
    
    indices = [problem["materials"][label] for label in problem["labels"]]
    
    R = _compute_reflectance(wavelengths_nm, indices, thicknesses, 
                             problem["ambient_index"], problem["substrate_index"])
    
    return np.mean(R) * 100.0

def _nelder_mead(f, x0, bounds, max_iter=2000, tol=1e-10, time_limit=50.0, start_time=None):
    """
    Custom Nelder-Mead simplex algorithm with bounds handling.
    """
    if start_time is None:
        start_time = time.time()
    
    n = len(x0)
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    
    for i in range(n):
        simplex[i + 1] = x0.copy()
        step = 0.1 * (bounds[i][1] - bounds[i][0]) if bounds[i][1] > bounds[i][0] else 0.1
        simplex[i + 1][i] += step
    
    fvals = np.array([f(simplex[i], bounds, start_time) for i in range(n + 1)])
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        idx = np.argsort(fvals)
        simplex = simplex[idx]
        fvals = fvals[idx]
        
        if np.max(fvals) - np.min(fvals) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        x_r = centroid + alpha * (centroid - simplex[-1])
        x_r = np.clip(x_r, [b[0] for b in bounds], [b[1] for b in bounds])
        f_r = f(x_r, bounds, start_time)
        
        if f_r < fvals[0]:
            x_e = centroid + gamma * (x_r - centroid)
            x_e = np.clip(x_e, [b[0] for b in bounds], [b[1] for b in bounds])
            f_e = f(x_e, bounds, start_time)
            
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
            x_c = np.clip(x_c, [b[0] for b in bounds], [b[1] for b in bounds])
            f_c = f(x_c, bounds, start_time)
            
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], [b[0] for b in bounds], [b[1] for b in bounds])
                    fvals[i] = f(simplex[i], bounds, start_time)
    
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]

def design(problem):
    """
    Design a broadband antireflection coating.
    Returns: (labels, thicknesses)
    """
    start_time = time.time()
    time_limit = 145.0
    
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wavelengths_nm = np.linspace(band_nm[0], band_nm[1], 401)
    
    best_result = None
    best_score = np.inf
    
    # Known good structures for broadband AR coatings
    # These are based on literature: alternating L/H layers with quarter-wave thicknesses
    # at different design wavelengths
    
    center_wl = 0.5 * (band_nm[0] + band_nm[1])  # 600 nm
    
    # Define several initial structures with different layer counts and patterns
    # Pattern 1: 9 layers (L H L H L H L H L) - common broadband AR
    # Pattern 2: 11 layers (L H L H L H L H L H L)
    # Pattern 3: 7 layers (L H L H L H L)
    # Pattern 4: 13 layers (L H L H L H L H L H L H L)
    # Pattern 5: 5 layers (L H L H L)
    
    structures = []
    
    # 9-layer structure: quarter-wave at multiple wavelengths
    n_layers_9 = 9
    labels_9 = ['L' if i % 2 == 0 else 'H' for i in range(n_layers_9)]
    indices_9 = [materials[label] for label in labels_9]
    # Use quarter-wave at 600nm for all layers as initial guess
    init_9 = np.array([600.0 / (4.0 * n) for n in indices_9])
    structures.append((labels_9, init_9))
    
    # 11-layer structure
    n_layers_11 = 11
    labels_11 = ['L' if i % 2 == 0 else 'H' for i in range(n_layers_11)]
    indices_11 = [materials[label] for label in labels_11]
    init_11 = np.array([600.0 / (4.0 * n) for n in indices_11])
    structures.append((labels_11, init_11))
    
    # 7-layer structure
    n_layers_7 = 7
    labels_7 = ['L' if i % 2 == 0 else 'H' for i in range(n_layers_7)]
    indices_7 = [materials[label] for label in labels_7]
    init_7 = np.array([600.0 / (4.0 * n) for n in indices_7])
    structures.append((labels_7, init_7))
    
    # 13-layer structure
    n_layers_13 = 13
    labels_13 = ['L' if i % 2 == 0 else 'H' for i in range(n_layers_13)]
    indices_13 = [materials[label] for label in labels_13]
    init_13 = np.array([600.0 / (4.0 * n) for n in indices_13])
    structures.append((labels_13, init_13))
    
    # 5-layer structure
    n_layers_5 = 5
    labels_5 = ['L' if i % 2 == 0 else 'H' for i in range(n_layers_5)]
    indices_5 = [materials[label] for label in labels_5]
    init_5 = np.array([600.0 / (4.0 * n) for n in indices_5])
    structures.append((labels_5, init_5))
    
    # Also try structures with different design wavelengths
    # Quarter-wave at 500nm
    for wl_design in [500.0, 550.0, 650.0, 700.0]:
        n_layers = 9
        labels = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        indices = [materials[label] for label in labels]
        init = np.array([wl_design / (4.0 * n) for n in indices])
        structures.append((labels, init))
    
    # Try a few random perturbations of the best initial structures
    np.random.seed(42)
    for labels, init in structures[:3]:
        for _ in range(3):
            perturbed = init * (1.0 + np.random.uniform(-0.2, 0.2, size=len(init)))
            perturbed = np.clip(perturbed, 0, max_layer_nm)
            structures.append((labels, perturbed))
    
    # Evaluate each structure with Nelder-Mead optimization
    for labels, init_thicknesses in structures:
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed - 5.0
        
        if remaining < 2.0:
            break
        
        n_layers = len(labels)
        if n_layers > max_layers:
            continue
        
        problem_local = dict(problem)
        problem_local["labels"] = labels
        
        bounds = [(0, max_layer_nm)] * n_layers
        
        def obj_func(x, bounds_local, start_time_local):
            return _objective(x, problem_local, wavelengths_nm)
        
        # Allocate time for this structure
        nm_time = min(remaining * 0.8, 25.0)
        
        try:
            x_opt, f_opt = _nelder_mead(obj_func, init_thicknesses, bounds,
                                         max_iter=1000, tol=1e-10,
                                         time_limit=nm_time,
                                         start_time=start_time)
            
            # Check constraints
            x_opt = np.clip(x_opt, 0, max_layer_nm)
            if np.sum(x_opt) > max_total_nm:
                x_opt = x_opt * (max_total_nm / np.sum(x_opt))
            
            # Re-evaluate to get accurate score
            f_opt = obj_func(x_opt, bounds, start_time)
            
            if f_opt < best_score:
                best_score = f_opt
                best_result = (labels, x_opt.tolist())
                
        except Exception:
            continue
    
    # Fallback
    if best_result is None:
        return ["L"], [center_wl / (4.0 * materials["L"])]
    
    labels, thicknesses = best_result
    
    # Ensure constraints
    thicknesses = np.clip(thicknesses, 0, max_layer_nm)
    if np.sum(thicknesses) > max_total_nm:
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
    
    return labels, thicknesses.tolist()
