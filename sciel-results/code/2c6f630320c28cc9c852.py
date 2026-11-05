import numpy as np
from scipy.optimize import minimize
import time

def _compute_reflectance(wavelengths_nm, n_ambient, n_substrate, materials, layer_labels, thicknesses_nm):
    """
    Compute normal-incidence reflectance using the characteristic matrix method.
    Returns reflectance as a fraction (0 to 1).
    """
    # Wavelengths in nm, convert to phase
    # For each wavelength, compute the transfer matrix
    n_L = materials['L']
    n_H = materials['H']
    
    # Precompute for speed
    n_layers = len(layer_labels)
    n_wl = len(wavelengths_nm)
    
    # Build the stack matrix for all wavelengths at once
    # M = product of individual layer matrices
    # Each layer matrix: [[cos(delta), i*sin(delta)/n], [i*n*sin(delta), cos(delta)]]
    # where delta = 2*pi*n*d/lambda
    
    # Initialize the product matrix as identity
    M_real = np.ones((n_wl, 2, 2), dtype=np.complex128)
    M_imag = np.zeros((n_wl, 2, 2), dtype=np.complex128)
    M = M_real + 1j * M_imag
    
    # Multiply from substrate side to air side (reverse order)
    for i in range(n_layers - 1, -1, -1):
        label = layer_labels[i]
        if label == 'L':
            n = n_L
        else:
            n = n_H
        d = thicknesses_nm[i]
        
        # delta = 2*pi*n*d/lambda
        delta = 2.0 * np.pi * n * d / wavelengths_nm  # shape (n_wl,)
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        # Layer matrix elements
        m11 = cos_d
        m12 = 1j * sin_d / n
        m21 = 1j * n * sin_d
        m22 = cos_d
        
        # M = M @ layer_matrix (matrix multiply for each wavelength)
        # M is (n_wl, 2, 2), layer is (n_wl, 2, 2)
        new_M = np.empty_like(M)
        new_M[:, 0, 0] = M[:, 0, 0] * m11 + M[:, 0, 1] * m21
        new_M[:, 0, 1] = M[:, 0, 0] * m12 + M[:, 0, 1] * m22
        new_M[:, 1, 0] = M[:, 1, 0] * m11 + M[:, 1, 1] * m21
        new_M[:, 1, 1] = M[:, 1, 0] * m12 + M[:, 1, 1] * m22
        M = new_M
    
    # Now compute reflectance
    # r = (n_ambient * M11 + n_ambient * n_substrate * M12 - n_substrate * M21 - n_substrate * M22) /
    #     (n_ambient * M11 + n_ambient * n_substrate * M12 + n_substrate * M21 + n_substrate * M22)
    # Actually, the standard formula:
    # Y = (M11*n_sub + M12) / (M21*n_sub + M22)  (admittance)
    # r = (n_ambient - Y) / (n_ambient + Y)
    # R = |r|^2
    
    M11 = M[:, 0, 0]
    M12 = M[:, 0, 1]
    M21 = M[:, 1, 0]
    M22 = M[:, 1, 1]
    
    Y = (M11 * n_substrate + M12) / (M21 * n_substrate + M22)
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.abs(r) ** 2
    
    return R

def _objective(thicknesses, n_ambient, n_substrate, materials, layer_labels, wavelengths_nm, max_layer_nm):
    """Objective: mean reflectance in percent, with penalty for constraint violations."""
    # Check constraints
    if any(t < 0 for t in thicknesses) or any(t > max_layer_nm for t in thicknesses):
        return 100.0 + 1000.0 * max(0, max(thicknesses) - max_layer_nm)
    total = sum(thicknesses)
    if total > 1000.0:
        return 100.0 + 1000.0 * (total - 1000.0)
    
    R = _compute_reflectance(wavelengths_nm, n_ambient, n_substrate, materials, layer_labels, thicknesses)
    return np.mean(R) * 100.0  # in percent

def design(problem):
    n_ambient = problem['ambient_index']
    n_substrate = problem['substrate_index']
    materials = problem['materials']
    band = problem['band_nm']
    max_layers = problem['max_layers']
    max_total_nm = problem['max_total_nm']
    max_layer_nm = problem['max_layer_nm']
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Wavelengths for evaluation (401 points)
    wavelengths_nm = np.linspace(band[0], band[1], 401)
    
    # Choose number of layers: use 10 layers for good broadband performance
    n_layers = 10
    if n_layers > max_layers:
        n_layers = max_layers
    
    # Initial guess: quarter-wave layers alternating L and H, centered at 600 nm
    center_wl = 0.5 * (band[0] + band[1])
    initial_thick = []
    for i in range(n_layers):
        if i % 2 == 0:
            n = materials['L']
        else:
            n = materials['H']
        d = center_wl / (4.0 * n)
        initial_thick.append(min(d, max_layer_nm))
    
    layer_labels = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
    
    # Evaluate initial solution
    R_init = _compute_reflectance(wavelengths_nm, n_ambient, n_substrate, materials, layer_labels, initial_thick)
    best_score = np.mean(R_init) * 100.0
    best_thick = list(initial_thick)
    
    # Try a few different layer counts and patterns
    candidate_configs = []
    
    # Config 1: 10 layers, L-H-L-H...
    candidate_configs.append((10, ['L' if i % 2 == 0 else 'H' for i in range(10)]))
    
    # Config 2: 12 layers
    candidate_configs.append((12, ['L' if i % 2 == 0 else 'H' for i in range(12)]))
    
    # Config 3: 8 layers
    candidate_configs.append((8, ['L' if i % 2 == 0 else 'H' for i in range(8)]))
    
    # Config 4: 14 layers
    candidate_configs.append((14, ['L' if i % 2 == 0 else 'H' for i in range(14)]))
    
    # Config 5: 16 layers
    candidate_configs.append((16, ['L' if i % 2 == 0 else 'H' for i in range(16)]))
    
    # Config 6: Start with H
    candidate_configs.append((10, ['H' if i % 2 == 0 else 'L' for i in range(10)]))
    
    # Config 7: 12 layers starting with H
    candidate_configs.append((12, ['H' if i % 2 == 0 else 'L' for i in range(12)]))
    
    best_result = None
    best_mean_R = float('inf')
    
    for n_l, labels in candidate_configs:
        if time.time() > deadline:
            break
        if n_l > max_layers:
            continue
        
        # Initial guess: quarter-wave at center
        init = []
        for i in range(n_l):
            if labels[i] == 'L':
                n = materials['L']
            else:
                n = materials['H']
            d = center_wl / (4.0 * n)
            init.append(min(d, max_layer_nm))
        
        # Evaluate
        R = _compute_reflectance(wavelengths_nm, n_ambient, n_substrate, materials, labels, init)
        mean_R = np.mean(R) * 100.0
        if mean_R < best_mean_R:
            best_mean_R = mean_R
            best_result = (labels, list(init))
        
        # Optimize with Nelder-Mead
        try:
            def obj(x):
                if time.time() > deadline:
                    return 1000.0
                return _objective(x, n_ambient, n_substrate, materials, labels, wavelengths_nm, max_layer_nm)
            
            res = minimize(obj, init, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 0.01, 'fatol': 0.001, 'adaptive': True})
            
            if res.fun < best_mean_R and res.fun < 100.0:
                # Check constraints
                if all(0 <= t <= max_layer_nm for t in res.x) and sum(res.x) <= max_total_nm:
                    best_mean_R = res.fun
                    best_result = (labels, list(res.x))
        except Exception:
            pass
    
    # If no result found, use the initial single-layer solution
    if best_result is None:
        return ["L"], [center_wl / (4.0 * materials['L'])]
    
    labels, thick = best_result
    # Final constraint check
    thick = [max(0.0, min(t, max_layer_nm)) for t in thick]
    total = sum(thick)
    if total > max_total_nm:
        scale = max_total_nm / total
        thick = [t * scale for t in thick]
    
    return labels, thick
