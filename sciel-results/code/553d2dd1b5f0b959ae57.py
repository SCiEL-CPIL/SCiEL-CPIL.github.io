"""Broadband antireflection coating design via multi-start local search with gradient descent."""
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
    N = len(wavelengths_nm)
    
    M00 = np.ones(N, dtype=complex)
    M01 = np.zeros(N, dtype=complex)
    M10 = np.zeros(N, dtype=complex)
    M11 = np.ones(N, dtype=complex)
    
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

def _gradient_descent(f, x0, bounds, start_time, time_budget=10.0, lr=0.05, tol=1e-8):
    """
    Gradient descent with finite difference gradients.
    """
    if time.time() - start_time > start_time + time_budget:
        return x0, f(x0, bounds, start_time)
    
    n = len(x0)
    x = x0.copy()
    best_x = x.copy()
    best_f = f(x, bounds, start_time)
    
    for it in range(500):
        if time.time() - start_time > start_time + time_budget:
            break
        
        # Compute gradient via finite differences
        grad = np.zeros(n)
        for i in range(n):
            h = 0.01
            x_plus = x.copy()
            x_plus[i] = min(x_plus[i] + h, bounds[i][1])
            x_minus = x.copy()
            x_minus[i] = max(x_minus[i] - h, bounds[i][0])
            
            f_plus = f(x_plus, bounds, start_time)
            f_minus = f(x_minus, bounds, start_time)
            
            if x_plus[i] - x_minus[i] > 0:
                grad[i] = (f_plus - f_minus) / (x_plus[i] - x_minus[i])
        
        # Check convergence
        if np.max(np.abs(grad)) < tol:
            break
        
        # Update
        x_new = x - lr * grad
        x_new = np.clip(x_new, [b[0] for b in bounds], [b[1] for b in bounds])
        
        f_new = f(x_new, bounds, start_time)
        
        if f_new < best_f:
            best_f = f_new
            best_x = x_new.copy()
        
        # Adaptive learning rate
        if f_new > best_f * 1.01:
            lr *= 0.5
        elif f_new < best_f * 0.99:
            lr = min(lr * 1.1, 0.5)
        
        x = x_new
    
    return best_x, best_f

def design(problem):
    """
    Design a broadband antireflection coating.
    Returns: (labels, thicknesses)
    """
    start_time = time.time()
    time_limit = 140.0
    
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
    
    # Try different numbers of layers
    layer_counts = [4, 6, 8, 10, 12, 14, 16, 18]
    
    for n_layers in layer_counts:
        if time.time() - start_time > time_limit - 15:
            break
        
        if n_layers > max_layers:
            break
        
        # Alternate L, H starting with L
        labels = []
        for i in range(n_layers):
            if i % 2 == 0:
                labels.append('L')
            else:
                labels.append('H')
        
        problem_local = dict(problem)
        problem_local["labels"] = labels
        
        indices = [materials[label] for label in labels]
        bounds = [(0, max_layer_nm)] * n_layers
        
        def obj_func(x, bounds_local, start_time_local):
            return _objective(x, problem_local, wavelengths_nm)
        
        # Multiple random starts
        n_starts = 5
        for start_i in range(n_starts):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Initial guess: quarter-wave at center wavelength with some randomness
            center_wl = 0.5 * (band_nm[0] + band_nm[1])
            if start_i == 0:
                x0 = np.array([center_wl / (4.0 * n) for n in indices])
            else:
                x0 = np.random.uniform(0, max_layer_nm, n_layers)
            
            # Phase 1: Gradient descent
            remaining = time_limit - (time.time() - start_time) - 5
            gd_time = min(8.0, remaining * 0.4)
            try:
                x_gd, f_gd = _gradient_descent(obj_func, x0, bounds, start_time, 
                                                time_budget=gd_time, lr=0.05, tol=1e-8)
            except Exception:
                x_gd = x0
                f_gd = obj_func(x0, bounds, start_time)
            
            # Phase 2: Nelder-Mead polish
            remaining = time_limit - (time.time() - start_time) - 3
            nm_time = min(5.0, remaining * 0.4)
            try:
                x_nm, f_nm = _nelder_mead(obj_func, x_gd, bounds, 
                                            max_iter=300, tol=1e-10, 
                                            time_limit=nm_time,
                                            start_time=start_time)
            except Exception:
                x_nm = x_gd
                f_nm = f_gd
            
            if f_nm < best_score:
                best_score = f_nm
                best_result = (labels, x_nm.tolist())
        
        # Early stop if very good
        if best_score < 0.3:
            break
    
    if best_result is None:
        center_wl = 0.5 * (band_nm[0] + band_nm[1])
        return ["L"], [center_wl / (4.0 * materials["L"])]
    
    labels, thicknesses = best_result
    thicknesses = np.clip(thicknesses, 0, max_layer_nm)
    if np.sum(thicknesses) > max_total_nm:
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
    
    return labels, thicknesses.tolist()
