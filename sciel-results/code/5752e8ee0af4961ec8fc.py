"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _reflectance_spectrum(wavelengths, indices, thicknesses, ambient, substrate):
    """
    Compute normal-incidence reflectance (fraction) for a multi-layer stack
    using the characteristic-matrix method.
    
    Parameters:
    - wavelengths: array of wavelengths in nm
    - indices: array of refractive indices for each layer
    - thicknesses: array of physical thicknesses in nm
    - ambient: ambient refractive index
    - substrate: substrate refractive index
    
    Returns:
    - reflectance: array of reflectance fractions
    """
    n = len(indices)
    # Convert wavelength to radians phase: delta = 2*pi*n*d/lambda
    # We work in complex matrices for generality, but here everything is real.
    # M_i = [[cos(delta_i), i*sin(delta_i)/n_i], [i*n_i*sin(delta_i), cos(delta_i)]]
    # Actually, the standard characteristic matrix for a layer with index n_i and thickness d_i:
    # M = [[cos(delta), i*sin(delta)/n_i], [i*n_i*sin(delta), cos(delta)]]
    # where delta = 2*pi*n_i*d_i/lambda
    
    # Total matrix M = M_1 * M_2 * ... * M_n
    # Then b = M[0,1], c = M[1,1]
    # Y = (ambient * c + b) / (ambient + c * b) ... wait, let me recall the exact formula.
    
    # The admittance Y = (A + B) / (C + D) where [[A,B],[C,D]] is the total matrix.
    # Actually, for normal incidence:
    # r = (Y_ambient - Y) / (Y_ambient + Y)
    # where Y_ambient = ambient (since ambient is real and normal incidence)
    # and Y = (A + B) / (C + D) ... no.
    
    # Let me use the standard formulation:
    # The total characteristic matrix M = [[m11, m12], [m21, m22]]
    # The optical admittance Y = (m11 + m12 * ambient) / (m21 + m22 * ambient) ... 
    # Wait, I need to be careful. Let me derive or recall correctly.
    
    # Standard thin-film theory:
    # For a stack of layers, the total characteristic matrix is the product of individual matrices.
    # For layer j: M_j = [[cos(delta_j), i*sin(delta_j)/n_j], [i*n_j*sin(delta_j), cos(delta_j)]]
    # Total matrix M = M_1 * M_2 * ... * M_N
    # 
    # The reflectance at normal incidence:
    # r = (m21 + m22 * n_s - m11 * n_0 - m12 * n_0 * n_s) / (m21 + m22 * n_s + m11 * n_0 + m12 * n_0 * n_s)
    # where n_0 is ambient, n_s is substrate.
    
    # Let me verify with a single layer:
    # M = [[cos(d), i*sin(d)/n], [i*n*sin(d), cos(d)]]
    # r = (i*n*sin(d) + cos(d)*n_s - cos(d)*n_0 - i*sin(d)/n * n_0 * n_s) / (i*n*sin(d) + cos(d)*n_s + cos(d)*n_0 + i*sin(d)/n * n_0 * n_s)
    # This is the standard formula. Let's use this.
    
    # Vectorize over wavelengths
    # delta_j = 2*pi*n_j*d_j / lambda
    # For each wavelength, compute the product of matrices.
    
    # Since we have at most 20 layers, and 401 wavelengths, we can vectorize.
    # Let's compute for all wavelengths at once.
    
    # wavelengths: shape (W,)
    # indices: shape (N,)
    # thicknesses: shape (N,)
    
    # delta: shape (W, N)
    delta = 2.0 * np.pi * np.outer(indices, thicknesses) / wavelengths[:, None]
    # delta[i, j] = 2*pi*n_j*d_j / lambda_i
    
    cos_d = np.cos(delta)  # shape (W, N)
    sin_d = np.sin(delta)  # shape (W, N)
    
    # For each layer j and wavelength i:
    # m11 = cos(delta)
    # m12 = i*sin(delta)/n_j
    # m21 = i*n_j*sin(delta)
    # m22 = cos(delta)
    
    # We need to multiply these matrices for all layers.
    # This is a bit tricky to vectorize fully, but we can do it iteratively over layers.
    
    # Initialize total matrix for each wavelength: identity
    W = len(wavelengths)
    m11 = np.ones(W, dtype=complex)
    m12 = np.zeros(W, dtype=complex)
    m21 = np.zeros(W, dtype=complex)
    m22 = np.ones(W, dtype=complex)
    
    for j in range(n):
        n_j = indices[j]
        # For this layer:
        # a = cos_d[:, j]
        # b = 1j * sin_d[:, j] / n_j
        # c = 1j * n_j * sin_d[:, j]
        # d = cos_d[:, j]
        
        a = cos_d[:, j]
        b = 1j * sin_d[:, j] / n_j
        c = 1j * n_j * sin_d[:, j]
        d = cos_d[:, j]
        
        # New total = old * layer_matrix
        # [m11, m12; m21, m22] * [a, b; c, d] = [m11*a + m12*c, m11*b + m12*d; m21*a + m22*c, m21*b + m22*d]
        
        new_m11 = m11 * a + m12 * c
        new_m12 = m11 * b + m12 * d
        new_m21 = m21 * a + m22 * c
        new_m22 = m21 * b + m22 * d
        
        m11 = new_m11
        m12 = new_m12
        m21 = new_m21
        m22 = new_m22
    
    # Now compute reflectance
    # r = (m21 + m22*n_s - m11*n_0 - m12*n_0*n_s) / (m21 + m22*n_s + m11*n_0 + m12*n_0*n_s)
    numerator = m21 + m22 * substrate - m11 * ambient - m12 * ambient * substrate
    denominator = m21 + m22 * substrate + m11 * ambient + m12 * ambient * substrate
    r = numerator / denominator
    reflectance = np.abs(r) ** 2
    
    return reflectance

def _mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance in percent."""
    wavelengths = np.linspace(problem["band_nm"][0], problem["band_nm"][1], 401)
    indices = np.array([problem["materials"][m] for m in materials])
    thicknesses = np.array(thicknesses, dtype=float)
    
    if len(thicknesses) == 0:
        # Bare substrate
        n_0 = problem["ambient_index"]
        n_s = problem["substrate_index"]
        r = (n_0 - n_s) / (n_0 + n_s)
        return 100.0 * r * r
    
    refl = _reflectance_spectrum(wavelengths, indices, thicknesses, problem["ambient_index"], problem["substrate_index"])
    return 100.0 * np.mean(refl)

def _is_valid(materials, thicknesses, problem):
    """Check if the design is valid."""
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    if len(materials) > max_layers:
        return False
    if len(materials) != len(thicknesses):
        return False
    for t in thicknesses:
        if t < 0 or t > max_layer_nm:
            return False
    if sum(thicknesses) > max_total_nm:
        return False
    return True

def _nelder_mead_optimize(materials, thicknesses, problem, time_limit, max_iter=5000):
    """Run Nelder-Mead optimization on the thicknesses."""
    start_time = time.time()
    
    def objective(x):
        x = np.clip(x, 0, problem.get("max_layer_nm", 400))
        if np.sum(x) > problem.get("max_total_nm", 1000):
            return 1e10
        return _mean_reflectance(materials, x, problem)
    
    initial = np.array(thicknesses, dtype=float)
    initial = np.clip(initial, 0, problem.get("max_layer_nm", 400))
    
    best_x = initial.copy()
    best_f = objective(initial)
    
    n = len(initial)
    # Create initial simplex
    simplex = np.zeros((n + 1, n))
    simplex[0] = initial
    for i in range(n):
        simplex[i + 1] = initial.copy()
        simplex[i + 1][i] += 0.1 if initial[i] < 0.1 else initial[i] * 0.1
    
    # Simple Nelder-Mead implementation
    alpha = 1.0  # reflection
    gamma = 2.0  # expansion
    rho = 0.5    # contraction
    sigma = 0.5  # shrink
    
    f_values = np.array([objective(simplex[i]) for i in range(n + 1)])
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Sort simplex by function value
        order = np.argsort(f_values)
        simplex = simplex[order]
        f_values = f_values[order]
        
        best = simplex[0]
        worst = simplex[n]
        second_worst = simplex[n - 1]
        
        centroid = np.mean(simplex[:n], axis=0)
        
        # Reflection
        x_r = centroid + alpha * (centroid - worst)
        x_r = np.clip(x_r, 0, problem.get("max_layer_nm", 400))
        f_r = objective(x_r)
        
        if f_values[0] <= f_r < f_values[n - 1]:
            simplex[n] = x_r
            f_values[n] = f_r
            continue
        
        # Expansion
        if f_r < f_values[0]:
            x_e = centroid + gamma * (x_r - centroid)
            x_e = np.clip(x_e, 0, problem.get("max_layer_nm", 400))
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[n] = x_e
                f_values[n] = f_e
            else:
                simplex[n] = x_r
                f_values[n] = f_r
            continue
        
        # Contraction
        if f_r < f_values[n]:
            x_c = centroid + rho * (worst - centroid)
            x_c = np.clip(x_c, 0, problem.get("max_layer_nm", 400))
            f_c = objective(x_c)
            if f_c <= f_r:
                simplex[n] = x_c
                f_values[n] = f_c
                continue
        
        # Shrink
        for i in range(1, n + 1):
            simplex[i] = best + sigma * (simplex[i] - best)
            f_values[i] = objective(simplex[i])
        
        # Track best
        min_idx = np.argmin(f_values)
        if f_values[min_idx] < best_f:
            best_f = f_values[min_idx]
            best_x = simplex[min_idx].copy()
    
    return best_x, best_f

def design(problem):
    """Design a broadband antireflection coating."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s wall-clock limit
    
    ambient = problem["ambient_index"]
    substrate = problem["substrate_index"]
    materials_dict = problem["materials"]
    band = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    centre_wl = 0.5 * (band[0] + band[1])
    
    # Try several layer configurations
    best_result = None
    best_score = float('inf')
    
    # Configuration 1: 2 layers (L-H or H-L)
    for config in [["L", "H"], ["H", "L"]]:
        indices = [materials_dict[m] for m in config]
        # Initial guess: quarter-wave at centre wavelength
        thicknesses = [centre_wl / (4.0 * n) for n in indices]
        
        if not _is_valid(config, thicknesses, problem):
            continue
        
        time_left = deadline - time.time()
        if time_left < 5:
            break
        
        x_opt, f_opt = _nelder_mead_optimize(config, thicknesses, problem, time_limit=time_left - 2)
        x_opt = np.clip(x_opt, 0, max_layer_nm)
        
        if _is_valid(config, x_opt, problem):
            if f_opt < best_score:
                best_score = f_opt
                best_result = (config, x_opt.tolist())
    
    # Configuration 2: 4 layers (L-H-L-H or H-L-H-L)
    for config in [["L", "H", "L", "H"], ["H", "L", "H", "L"]]:
        indices = [materials_dict[m] for m in config]
        thicknesses = [centre_wl / (4.0 * n) for n in indices]
        
        if not _is_valid(config, thicknesses, problem):
            continue
        
        time_left = deadline - time.time()
        if time_left < 10:
            break
        
        x_opt, f_opt = _nelder_mead_optimize(config, thicknesses, problem, time_limit=time_left - 2)
        x_opt = np.clip(x_opt, 0, max_layer_nm)
        
        if _is_valid(config, x_opt, problem):
            if f_opt < best_score:
                best_score = f_opt
                best_result = (config, x_opt.tolist())
    
    # Configuration 3: 6 layers
    for config in [["L", "H", "L", "H", "L", "H"], ["H", "L", "H", "L", "H", "L"]]:
        indices = [materials_dict[m] for m in config]
        thicknesses = [centre_wl / (4.0 * n) for n in indices]
        
        if not _is_valid(config, thicknesses, problem):
            continue
        
        time_left = deadline - time.time()
        if time_left < 15:
            break
        
        x_opt, f_opt = _nelder_mead_optimize(config, thicknesses, problem, time_limit=time_left - 2)
        x_opt = np.clip(x_opt, 0, max_layer_nm)
        
        if _is_valid(config, x_opt, problem):
            if f_opt < best_score:
                best_score = f_opt
                best_result = (config, x_opt.tolist())
    
    # Configuration 4: 8 layers
    for config in [["L", "H", "L", "H", "L", "H", "L", "H"], ["H", "L", "H", "L", "H", "L", "H", "L"]]:
        indices = [materials_dict[m] for m in config]
        thicknesses = [centre_wl / (4.0 * n) for n in indices]
        
        if not _is_valid(config, thicknesses, problem):
            continue
        
        time_left = deadline - time.time()
        if time_left < 20:
            break
        
        x_opt, f_opt = _nelder_mead_optimize(config, thicknesses, problem, time_limit=time_left - 2)
        x_opt = np.clip(x_opt, 0, max_layer_nm)
        
        if _is_valid(config, x_opt, problem):
            if f_opt < best_score:
                best_score = f_opt
                best_result = (config, x_opt.tolist())
    
    # Configuration 5: 10 layers
    for config in [["L", "H", "L", "H", "L", "H", "L", "H", "L", "H"], 
                   ["H", "L", "H", "L", "H", "L", "H", "L", "H", "L"]]:
        indices = [materials_dict[m] for m in config]
        thicknesses = [centre_wl / (4.0 * n) for n in indices]
        
        if not _is_valid(config, thicknesses, problem):
            continue
        
        time_left = deadline - time.time()
        if time_left < 25:
            break
        
        x_opt, f_opt = _nelder_mead_optimize(config, thicknesses, problem, time_limit=time_left - 2)
        x_opt = np.clip(x_opt, 0, max_layer_nm)
        
        if _is_valid(config, x_opt, problem):
            if f_opt < best_score:
                best_score = f_opt
                best_result = (config, x_opt.tolist())
    
    # Configuration 6: 12 layers
    for config in [["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H"], 
                   ["H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"]]:
        indices = [materials_dict[m] for m in config]
        thicknesses = [centre_wl / (4.0 * n) for n in indices]
        
        if not _is_valid(config, thicknesses, problem):
            continue
        
        time_left = deadline - time.time()
        if time_left < 30:
            break
        
        x_opt, f_opt = _nelder_mead_optimize(config, thicknesses, problem, time_limit=time_left - 2)
        x_opt = np.clip(x_opt, 0, max_layer_nm)
        
        if _is_valid(config, x_opt, problem):
            if f_opt < best_score:
                best_score = f_opt
                best_result = (config, x_opt.tolist())
    
    # Configuration 7: 14 layers
    for config in [["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H"], 
                   ["H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"]]:
        indices = [materials_dict[m] for m in config]
        thicknesses = [centre_wl / (4.0 * n) for n in indices]
        
        if not _is_valid(config, thicknesses, problem):
            continue
        
        time_left = deadline - time.time()
        if time_left < 35:
            break
        
        x_opt, f_opt = _nelder_mead_optimize(config, thicknesses, problem, time_limit=time_left - 2)
        x_opt = np.clip(x_opt, 0, max_layer_nm)
        
        if _is_valid(config, x_opt, problem):
            if f_opt < best_score:
                best_score = f_opt
                best_result = (config, x_opt.tolist())
    
    # Configuration 8: 16 layers
    for config in [["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H"], 
                   ["H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"]]:
        indices = [materials_dict[m] for m in config]
        thicknesses = [centre_wl / (4.0 * n) for n in indices]
        
        if not _is_valid(config, thicknesses, problem):
            continue
        
        time_left = deadline - time.time()
        if time_left < 40:
            break
        
        x_opt, f_opt = _nelder_mead_optimize(config, thicknesses, problem, time_limit=time_left - 2)
        x_opt = np.clip(x_opt, 0, max_layer_nm)
        
        if _is_valid(config, x_opt, problem):
            if f_opt < best_score:
                best_score = f_opt
                best_result = (config, x_opt.tolist())
    
    # Configuration 9: 18 layers
    for config in [["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H"], 
                   ["H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"]]:
        indices = [materials_dict[m] for m in config]
        thicknesses = [centre_wl / (4.0 * n) for n in indices]
        
        if not _is_valid(config, thicknesses, problem):
            continue
        
        time_left = deadline - time.time()
        if time_left < 45:
            break
        
        x_opt, f_opt = _nelder_mead_optimize(config, thicknesses, problem, time_limit=time_left - 2)
        x_opt = np.clip(x_opt, 0, max_layer_nm)
        
        if _is_valid(config, x_opt, problem):
            if f_opt < best_score:
                best_score = f_opt
                best_result = (config, x_opt.tolist())
    
    # Configuration 10: 20 layers
    for config in [["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H"], 
                   ["H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"]]:
        indices = [materials_dict[m] for m in config]
        thicknesses = [centre_wl / (4.0 * n) for n in indices]
        
        if not _is_valid(config, thicknesses, problem):
            continue
        
        time_left = deadline - time.time()
        if time_left < 50:
            break
        
        x_opt, f_opt = _nelder_mead_optimize(config, thicknesses, problem, time_limit=time_left - 2)
        x_opt = np.clip(x_opt, 0, max_layer_nm)
        
        if _is_valid(config, x_opt, problem):
            if f_opt < best_score:
                best_score = f_opt
                best_result = (config, x_opt.tolist())
    
    if best_result is not None:
        return best_result[0], best_result[1]
    
    # Fallback: single quarter-wave layer
    return ["L"], [centre_wl / (4.0 * materials_dict["L"])]
# EVOLVE-BLOCK-END
