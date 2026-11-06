"""Broadband antireflection coating design via vectorized characteristic-matrix evaluation and Nelder-Mead optimization."""
import numpy as np
import time
import math

def _compute_reflectance(wavelengths_nm, indices, thicknesses_nm, ambient_idx, substrate_idx):
    """
    Compute reflectance (fraction, 0-1) at each wavelength using the characteristic matrix method.
    wavelengths_nm: array of wavelengths in nm
    indices: list/array of refractive indices for each layer (air-side to substrate-side)
    thicknesses_nm: list/array of physical thicknesses in nm
    ambient_idx: refractive index of ambient (air)
    substrate_idx: refractive index of substrate
    Returns: array of reflectance values (0 to 1)
    """
    n_layers = len(indices)
    if n_layers == 0:
        # Bare substrate
        r0 = (ambient_idx - substrate_idx) / (ambient_idx + substrate_idx)
        return np.full(len(wavelengths_nm), r0 * r0)
    
    # Convert to optical path lengths: n * d
    optical_thick = np.array(thicknesses_nm) * np.array(indices)
    
    # k = 2*pi / lambda (in 1/nm)
    k = 2.0 * np.pi / wavelengths_nm  # shape (N,)
    
    # For each layer, compute B matrix: [[cos(delta), i*sin(delta)/n], [i*n*sin(delta), cos(delta)]]
    # delta = k * n * d = k * optical_thick
    
    # We'll compute the total characteristic matrix M = M1 * M2 * ... * Mn
    # Start with identity
    # M is 2x2, we'll use complex arrays
    # To vectorize over wavelengths, we represent M as a 4D array? No, let's do it differently.
    # Actually, for small number of layers (<=20) and moderate wavelengths (401), a loop over layers is fine.
    # But we can vectorize over wavelengths.
    
    # Let's represent the matrix as four arrays: M00, M01, M10, M11, each of shape (N,)
    M00 = np.ones(len(wavelengths_nm), dtype=complex)
    M01 = np.zeros(len(wavelengths_nm), dtype=complex)
    M10 = np.zeros(len(wavelengths_nm), dtype=complex)
    M11 = np.ones(len(wavelengths_nm), dtype=complex)
    
    for i in range(n_layers):
        n_i = indices[i]
        d_i = thicknesses_nm[i]
        delta = k * n_i * d_i  # shape (N,)
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        # Layer matrix:
        # B = [[cos(d), 1j*sin(d)/n], [1j*n*sin(d), cos(d)]]
        B00 = cos_d
        B01 = 1j * sin_d / n_i
        B10 = 1j * n_i * sin_d
        B11 = cos_d
        
        # New M = M * B
        # M_new[0,0] = M00*B00 + M01*B10
        # M_new[0,1] = M00*B01 + M01*B11
        # M_new[1,0] = M10*B00 + M11*B10
        # M_new[1,1] = M10*B01 + M11*B11
        
        new_M00 = M00 * B00 + M01 * B10
        new_M01 = M00 * B01 + M01 * B11
        new_M10 = M10 * B00 + M11 * B10
        new_M11 = M10 * B01 + M11 * B11
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    # Admittance: Y = (M10 + M11) / (M00 + M01)
    # Wait, the characteristic matrix relates (E, H) at the first interface to (E, H) at the last.
    # The input admittance is Y = (M10 + M11) / (M00 + M01) ... let me double-check.
    # Actually, the standard formula is:
    # [E]   [M00 M01] [E]
    # [H] = [M10 M11] [H]
    # So H_in = M10 * E_out + M11 * H_out
    # E_in = M00 * E_out + M01 * H_out
    # Y_in = H_in / E_in = (M10 * E_out + M11 * H_out) / (M00 * E_out + M01 * H_out)
    # = (M10 + M11 * Y_out) / (M00 + M01 * Y_out)
    # where Y_out = substrate_idx (since H_out/E_out = n_substrate for a wave going into the substrate)
    
    Y_out = substrate_idx
    Y_in = (M10 + M11 * Y_out) / (M00 + M01 * Y_out)
    
    # Reflectance at the first interface (ambient to coating):
    r = (ambient_idx - Y_in) / (ambient_idx + Y_in)
    R = np.abs(r) ** 2
    
    return R

def _objective(params, problem, wavelengths_nm):
    """
    Objective function: mean reflectance in percent.
    params: flat array of thicknesses
    Returns: mean reflectance * 100
    """
    max_layer_nm = problem.get("max_layer_nm", 400)
    max_total_nm = problem.get("max_total_nm", 1000)
    
    # Clip to valid range
    thicknesses = np.clip(params, 0, max_layer_nm)
    
    # Check total thickness constraint
    if np.sum(thicknesses) > max_total_nm:
        # Penalize heavily
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
    # Initialize simplex
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    
    # Create initial simplex with small perturbations
    for i in range(n):
        simplex[i + 1] = x0.copy()
        step = 0.1 * (bounds[i][1] - bounds[i][0]) if bounds[i][1] > bounds[i][0] else 0.1
        simplex[i + 1][i] += step
    
    # Evaluate initial simplex
    fvals = np.array([f(simplex[i], bounds, start_time) for i in range(n + 1)])
    
    alpha = 1.0  # reflection
    gamma = 2.0  # expansion
    rho = 0.5   # contraction
    sigma = 0.5  # shrink
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Sort simplex by fvals
        idx = np.argsort(fvals)
        simplex = simplex[idx]
        fvals = fvals[idx]
        
        # Check convergence
        if np.max(fvals) - np.min(fvals) < tol:
            break
        
        # Centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = centroid + alpha * (centroid - simplex[-1])
        x_r = np.clip(x_r, [b[0] for b in bounds], [b[1] for b in bounds])
        f_r = f(x_r, bounds, start_time)
        
        if f_r < fvals[0]:
            # Expansion
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
            # Contraction
            x_c = centroid + rho * (simplex[-1] - centroid)
            x_c = np.clip(x_c, [b[0] for b in bounds], [b[1] for b in bounds])
            f_c = f(x_c, bounds, start_time)
            
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], [b[0] for b in bounds], [b[1] for b in bounds])
                    fvals[i] = f(simplex[i], bounds, start_time)
    
    # Return best point
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]

def _polish(x0, bounds, f, start_time, time_budget=10.0, tol=1e-12):
    """
    Final high-precision polish pass using tight-tolerance Nelder-Mead.
    """
    if time.time() - start_time > start_time + time_budget:
        return x0, f(x0, bounds, start_time)
    
    n = len(x0)
    # Initialize simplex with very small perturbations
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    
    for i in range(n):
        simplex[i + 1] = x0.copy()
        step = 0.01 * (bounds[i][1] - bounds[i][0]) if bounds[i][1] > bounds[i][0] else 0.01
        simplex[i + 1][i] += step
    
    fvals = np.array([f(simplex[i], bounds, start_time) for i in range(n + 1)])
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    max_iter = 500
    for it in range(max_iter):
        if time.time() - start_time > start_time + time_budget:
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
    time_limit = 140.0  # Safety margin before 150s kill
    
    # Problem parameters
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    # Wavelengths for evaluation
    wavelengths_nm = np.linspace(band_nm[0], band_nm[1], 401)
    
    # Try different numbers of layers and optimize
    best_result = None
    best_score = np.inf
    best_x = None
    best_labels = None
    best_bounds = None
    
    # Strategy: try odd numbers of layers, starting with higher counts that
    # are known to perform better for broadband AR
    layer_counts = [9, 11, 13, 15, 17, 19, 7, 5, 3]
    
    for n_layers in layer_counts:
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed - 15.0  # Reserve 15s for polish
        
        if remaining < 3.0:
            break
        
        if n_layers > max_layers:
            continue
        
        # Create labels: alternate L, H, L, H, ... starting with L
        labels = []
        for i in range(n_layers):
            if i % 2 == 0:
                labels.append('L')
            else:
                labels.append('H')
        
        problem_local = dict(problem)
        problem_local["labels"] = labels
        
        # Initial guess: quarter-wave thickness at center wavelength
        center_wl = 0.5 * (band_nm[0] + band_nm[1])
        indices = [materials[label] for label in labels]
        init_thicknesses = np.array([center_wl / (4.0 * n) for n in indices])
        
        # Bounds
        bounds = [(0, max_layer_nm)] * n_layers
        
        # Objective function wrapper
        def obj_func(x, bounds_local, start_time_local):
            return _objective(x, problem_local, wavelengths_nm)
        
        # Adaptive time allocation: give more time to early (higher layer count) trials
        # but cap at a reasonable amount
        trial_time = min(remaining * 0.4, 20.0)
        
        # Run Nelder-Mead
        try:
            x_opt, f_opt = _nelder_mead(obj_func, init_thicknesses, bounds, 
                                         max_iter=800, tol=1e-8, 
                                         time_limit=trial_time,
                                         start_time=start_time)
        except Exception:
            continue
        
        if f_opt < best_score:
            best_score = f_opt
            best_result = (labels, x_opt.tolist())
            best_x = x_opt.copy()
            best_labels = labels.copy()
            best_bounds = bounds
            best_problem_local = problem_local
        
        # If we've done well, we can stop early
        if best_score < 0.8:
            break
    
    # If no result found, fall back to single layer
    if best_result is None:
        center_wl = 0.5 * (band_nm[0] + band_nm[1])
        return ["L"], [center_wl / (4.0 * materials["L"])]
    
    # Final polish pass on the best result
    elapsed = time.time() - start_time
    polish_budget = min(15.0, time_limit - elapsed - 3.0)
    
    if polish_budget > 2.0 and best_x is not None:
        try:
            x_polished, f_polished = _polish(best_x, best_bounds, 
                                              lambda x, b, st: _objective(x, best_problem_local, wavelengths_nm),
                                              start_time, time_budget=polish_budget)
            if f_polished < best_score:
                best_score = f_polished
                best_result = (best_labels, x_polished.tolist())
        except Exception:
            pass
    
    labels, thicknesses = best_result
    
    # Ensure constraints are satisfied
    thicknesses = np.clip(thicknesses, 0, max_layer_nm)
    if np.sum(thicknesses) > max_total_nm:
        # Scale down proportionally
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
    
    return labels, thicknesses.tolist()
