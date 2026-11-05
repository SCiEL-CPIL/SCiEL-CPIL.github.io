import time
import math
import numpy as np
import random
from scipy.optimize import minimize

def _reflectance_array(materials, thicknesses, n_ambient, n_substrate, wls, n_mat):
    """
    Compute reflectance array for given materials and thicknesses using
    the thin-film characteristic-matrix method.
    Returns array of reflectance values (0 to 1).
    """
    if len(materials) == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return np.full(len(wls), r * r)

    n_layers = len(materials)
    n_wls = len(wls)
    
    # Precompute beta = 2*pi*n*d/wl for all layers and wavelengths
    beta = np.zeros((n_layers, n_wls))
    for i, m in enumerate(materials):
        n = n_mat[m]
        d = thicknesses[i]
        if d == 0:
            continue
        beta[i, :] = 2.0 * math.pi * n * d / wls

    cosb = np.cos(beta)
    sinb = np.sin(beta)

    # Build the total characteristic matrix
    M11 = np.ones(n_wls, dtype=complex)
    M12 = np.zeros(n_wls, dtype=complex)
    M21 = np.zeros(n_wls, dtype=complex)
    M22 = np.ones(n_wls, dtype=complex)

    for i, m in enumerate(materials):
        n = n_mat[m]
        c = cosb[i]
        s = sinb[i]
        
        # Matrix elements for layer i
        A = c
        B = 1j * s / n
        C = 1j * n * s
        D = c

        # Matrix multiplication: M_new = M_old * M_i
        nM11 = M11 * A + M12 * C
        nM12 = M11 * B + M12 * D
        nM21 = M21 * A + M22 * C
        nM22 = M21 * B + M22 * D
        
        M11, M12, M21, M22 = nM11, nM12, nM21, nM22

    # Admittance Y
    Y = (M21 + n_substrate * M22) / (M11 + n_substrate * M12)
    
    # Reflectance r
    r = (n_ambient - Y) / (n_ambient + Y)
    
    # Reflectance R = |r|^2
    R = np.abs(r) ** 2
    return R

def _mean_reflectance_percent(materials, thicknesses, problem, wls, n_mat):
    """
    Compute mean reflectance in percent.
    Returns penalty if constraints are violated.
    """
    if len(materials) != len(thicknesses):
        return 1e6
    if len(materials) > problem["max_layers"]:
        return 1e6
        
    max_layer = problem["max_layer_nm"]
    max_total = problem["max_total_nm"]
    
    for t in thicknesses:
        if t < -1e-9 or t > max_layer + 1e-9:
            return 1e6
            
    total_th = sum(thicknesses)
    if total_th > max_total + 1e-6:
        return 1e6 + (total_th - max_total) * 10.0

    if len(materials) == 0:
        r = (problem["ambient_index"] - problem["substrate_index"]) / (
            problem["ambient_index"] + problem["substrate_index"])
        return r * r * 100.0

    R = _reflectance_array(materials, thicknesses, problem["ambient_index"],
                           problem["substrate_index"], wls, n_mat)
    return float(np.mean(R)) * 100.0

def _gradient_reflectance(materials, thicknesses, n_ambient, n_substrate, wls, n_mat):
    """
    Compute the gradient of mean reflectance with respect to thicknesses.
    Uses central differences for simplicity and reliability.
    """
    n_layers = len(materials)
    grad = np.zeros(n_layers)
    h = 0.1  # Step size for finite differences
    
    base_val = _mean_reflectance_percent(materials, thicknesses, 
                                          {'ambient_index': n_ambient, 'substrate_index': n_substrate,
                                           'max_layers': len(materials), 'max_total_nm': sum(thicknesses),
                                           'max_layer_nm': max(thicknesses) if thicknesses else 400},
                                          wls, n_mat)
    
    for i in range(n_layers):
        th_plus = list(thicknesses)
        th_minus = list(thicknesses)
        th_plus[i] += h
        th_minus[i] -= h
        
        # Clamp to valid range
        th_plus[i] = max(0, min(th_plus[i], 400))
        th_minus[i] = max(0, min(th_minus[i], 400))
        
        val_plus = _mean_reflectance_percent(materials, th_plus, 
                                              {'ambient_index': n_ambient, 'substrate_index': n_substrate,
                                               'max_layers': len(materials), 'max_total_nm': sum(th_plus),
                                               'max_layer_nm': max(th_plus) if th_plus else 400},
                                              wls, n_mat)
        val_minus = _mean_reflectance_percent(materials, th_minus, 
                                               {'ambient_index': n_ambient, 'substrate_index': n_substrate,
                                                'max_layers': len(materials), 'max_total_nm': sum(th_minus),
                                                'max_layer_nm': max(th_minus) if th_minus else 400},
                                               wls, n_mat)
        
        grad[i] = (val_plus - val_minus) / (2 * h)
    
    return grad

def design(problem):
    """
    Design a broadband antireflection coating.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    band_start, band_end = problem["band_nm"]
    wls = np.linspace(band_start, band_end, 401)
    n_mat = problem["materials"]
    
    best_val = float('inf')
    best_mats = None
    best_ths = None
    
    # Define candidate material sequences
    sequences = []
    
    # Length 2
    sequences.append(['L', 'H'])
    sequences.append(['H', 'L'])
    
    # Length 4
    sequences.append(['L', 'H', 'L', 'H'])
    sequences.append(['H', 'L', 'H', 'L'])
    sequences.append(['L', 'H', 'H', 'L'])
    sequences.append(['H', 'L', 'L', 'H'])
    
    # Length 6
    sequences.append(['L', 'H', 'L', 'H', 'L', 'H'])
    sequences.append(['H', 'L', 'H', 'L', 'H', 'L'])
    sequences.append(['L', 'H', 'H', 'L', 'H', 'L'])
    sequences.append(['H', 'L', 'L', 'H', 'L', 'H'])
    
    # Length 8
    sequences.append(['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'])
    sequences.append(['H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'])
    
    # Length 10
    sequences.append(['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'])
    sequences.append(['H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'])
    
    # Length 12
    sequences.append(['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'])
    sequences.append(['H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'])
    
    # Length 14
    sequences.append(['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'])
    sequences.append(['H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'])
    
    # Length 16
    sequences.append(['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'])
    sequences.append(['H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'])
    
    # Length 18
    sequences.append(['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'])
    sequences.append(['H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'])
    
    # Length 20
    sequences.append(['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'])
    sequences.append(['H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'])
    
    # Filter sequences by max_layers
    sequences = [seq for seq in sequences if len(seq) <= max_layers]
    
    # Generate initial guesses based on quarter-wave stacks
    center_wls = [450, 500, 550, 600, 650, 700, 525, 575, 625, 675, 475, 525, 575, 625, 675, 725]
    
    for mats in sequences:
        if time.time() > deadline - 5:
            break
            
        n_layers = len(mats)
        lo = np.zeros(n_layers)
        hi = np.full(n_layers, max_layer)
        
        # Generate initial guesses
        initial_guesses = []
        
        for cw in center_wls:
            ths = []
            for m in mats:
                n = n_mat[m]
                q = cw / (4.0 * n)
                q = q * random.uniform(0.85, 1.15)
                ths.append(q)
            initial_guesses.append(ths)
            
        # Add more diverse random guesses
        for _ in range(6):
            ths = []
            for _ in range(n_layers):
                if random.random() < 0.3:
                    ths.append(random.uniform(5, 80))
                elif random.random() < 0.6:
                    ths.append(random.uniform(40, 180))
                else:
                    ths.append(random.uniform(100, 300))
            initial_guesses.append(ths)
        
        # Add some structured guesses
        for scale in [0.5, 0.8, 1.0, 1.2, 1.5]:
            ths = []
            for i, m in enumerate(mats):
                n = n_mat[m]
                base = 550.0 / (4.0 * n)
                # Vary by position
                factor = 1.0 + 0.3 * math.sin(i * 0.7)
                ths.append(base * scale * factor)
            initial_guesses.append(ths)
        
        for x0 in initial_guesses:
            if time.time() > deadline - 5:
                break
                
            x0 = np.array(x0, dtype=float)
            x0 = np.clip(x0, lo, hi)
            # Scale to fit total thickness constraint if needed
            s = sum(x0)
            if s > max_total:
                x0 = x0 * (max_total / s)
                x0 = np.clip(x0, 0.0, max_layer)
            
            # Define objective for scipy
            def objective(th):
                th_list = list(np.clip(th, 0.0, max_layer))
                # Check total thickness
                if sum(th_list) > max_total + 1e-6:
                    return 1e6
                return _mean_reflectance_percent(mats, th_list, problem, wls, n_mat)
            
            # Define gradient
            def gradient(th):
                th_list = list(np.clip(th, 0.0, max_layer))
                if sum(th_list) > max_total + 1e-6:
                    return np.zeros(len(th))
                return _gradient_reflectance(mats, th_list, n_ambient, n_substrate, wls, n_mat)
            
            # Try L-BFGS-B with gradient
            try:
                result = minimize(objective, x0, method='L-BFGS-B', 
                                  jac=gradient, bounds=list(zip(lo, hi)),
                                  options={'maxiter': 200, 'gtol': 1e-6, 'maxfun': 500})
                
                x_opt = np.clip(result.x, 0.0, max_layer)
                s = sum(x_opt)
                if s > max_total:
                    x_opt = x_opt * (max_total / s)
                    x_opt = np.clip(x_opt, 0.0, max_layer)
                
                val = _mean_reflectance_percent(mats, list(x_opt), problem, wls, n_mat)
                
                if val < best_val:
                    best_val = val
                    best_mats = mats
                    best_ths = list(x_opt)
                    
            except Exception:
                pass
            
            # Also try Nelder-Mead from the same starting point if time permits
            if time.time() < deadline - 10:
                def nm_objective(th, m=mats):
                    th_list = list(np.clip(th, 0.0, max_layer))
                    if sum(th_list) > max_total + 1e-6:
                        return 1e6
                    return _mean_reflectance_percent(m, th_list, problem, wls, n_mat)
                
                try:
                    result_nm = minimize(nm_objective, x0, method='Nelder-Mead',
                                         bounds=list(zip(lo, hi)),
                                         options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-6})
                    
                    x_opt_nm = np.clip(result_nm.x, 0.0, max_layer)
                    s = sum(x_opt_nm)
                    if s > max_total:
                        x_opt_nm = x_opt_nm * (max_total / s)
                        x_opt_nm = np.clip(x_opt_nm, 0.0, max_layer)
                    
                    val_nm = _mean_reflectance_percent(mats, list(x_opt_nm), problem, wls, n_mat)
                    
                    if val_nm < best_val:
                        best_val = val_nm
                        best_mats = mats
                        best_ths = list(x_opt_nm)
                        
                except Exception:
                    pass
    
    # Final high-precision polish pass if time permits
    if best_mats is not None and time.time() < deadline - 15:
        n_layers = len(best_mats)
        lo = np.zeros(n_layers)
        hi = np.full(n_layers, max_layer)
        
        def objective(th):
            th_list = list(np.clip(th, 0.0, max_layer))
            if sum(th_list) > max_total + 1e-6:
                return 1e6
            return _mean_reflectance_percent(best_mats, th_list, problem, wls, n_mat)
        
        def gradient(th):
            th_list = list(np.clip(th, 0.0, max_layer))
            if sum(th_list) > max_total + 1e-6:
                return np.zeros(len(th))
            return _gradient_reflectance(best_mats, th_list, n_ambient, n_substrate, wls, n_mat)
        
        # Run a final polish with L-BFGS-B
        try:
            result_polish = minimize(objective, np.array(best_ths), method='L-BFGS-B',
                                     jac=gradient, bounds=list(zip(lo, hi)),
                                     options={'maxiter': 500, 'gtol': 1e-8, 'maxfun': 1000})
            
            x_polish = np.clip(result_polish.x, 0.0, max_layer)
            s = sum(x_polish)
            if s > max_total:
                x_polish = x_polish * (max_total / s)
                x_polish = np.clip(x_polish, 0.0, max_layer)
            
            val_polish = _mean_reflectance_percent(best_mats, list(x_polish), problem, wls, n_mat)
            
            if val_polish < best_val:
                best_val = val_polish
                best_ths = list(x_polish)
                
        except Exception:
            pass
    
    if best_mats is None:
        # Fallback: single layer
        best_mats = ['L']
        best_ths = [600.0 / (4.0 * n_L)]
        best_val = _mean_reflectance_percent(best_mats, best_ths, problem, wls, n_mat)
        
    # Final constraint check and scaling
    best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
    if sum(best_ths) > max_total + 1e-6:
        f = max_total / sum(best_ths)
        best_ths = [t * f for t in best_ths]
        
    return list(best_mats), list(best_ths)
