import time
import math
import numpy as np
import random
from scipy.optimize import minimize, Bounds

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

    ns = np.array([n_mat[m] for m in materials])
    ds = np.array(thicknesses)
    beta = 2.0 * math.pi * np.outer(ns * ds, 1.0 / wls)

    cosb = np.cos(beta)
    sinb = np.sin(beta)

    M11 = np.ones(n_wls, dtype=complex)
    M12 = np.zeros(n_wls, dtype=complex)
    M21 = np.zeros(n_wls, dtype=complex)
    M22 = np.ones(n_wls, dtype=complex)

    for i in range(n_layers):
        n = ns[i]
        c = cosb[i]
        s = sinb[i]

        A = c
        B = 1j * s / n
        C = 1j * n * s
        D = c

        nM11 = M11 * A + M12 * C
        nM12 = M11 * B + M12 * D
        nM21 = M21 * A + M22 * C
        nM22 = M21 * B + M22 * D

        M11, M12, M21, M22 = nM11, nM12, nM21, nM22

    Y = (M21 + n_substrate * M22) / (M11 + n_substrate * M12)
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.abs(r) ** 2
    return R

def _mean_reflectance_percent(materials, thicknesses, problem, wls, n_mat):
    """
    Compute mean reflectance in percent with constraint penalties.
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

def _project_to_feasible(x, max_layer, max_total):
    """Project x to feasible region: [0, max_layer] per layer, sum <= max_total."""
    x = np.clip(x, 0, max_layer)
    total = np.sum(x)
    if total > max_total:
        if total > 1e-12:
            x = x * (max_total / total)
        else:
            x = np.zeros_like(x)
    return x

def _optimize_with_lbfgs(objective, x0, lo, hi, deadline, max_iter=500):
    """
    Optimize using L-BFGS-B with constraints.
    """
    x0 = np.array(x0, dtype=float)
    n = len(x0)
    
    if n == 0:
        return x0, objective(x0)
    
    # Define bounds
    bounds = Bounds(lo, hi)
    
    # Add constraint for total thickness
    def constraint_total(x):
        return max_total - np.sum(x)
    
    constraints = [{'type': 'ineq', 'fun': constraint_total}]
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                         constraints=constraints, options={'maxiter': max_iter, 
                                                          'ftol': 1e-12, 
                                                          'gtol': 1e-8})
        if result.success or result.fun < objective(x0):
            x_opt = _project_to_feasible(result.x, hi[0], max_total)
            f_opt = objective(x_opt)
            return x_opt, f_opt
    except Exception:
        pass
    
    return x0, objective(x0)

def _optimize_stack(mats, ths, problem, wls, n_mat, deadline, max_layer, max_total):
    """
    Optimize a single stack using L-BFGS-B.
    """
    lo = np.zeros(len(ths))
    hi = np.full(len(ths), max_layer)
    
    def objective(th):
        th = _project_to_feasible(th, max_layer, max_total)
        return _mean_reflectance_percent(mats, list(th), problem, wls, n_mat)
    
    x0 = np.array(ths, dtype=float)
    x0 = _project_to_feasible(x0, max_layer, max_total)
    
    # Use L-BFGS-B for optimization
    if time.time() < deadline - 5:
        x_opt, f_opt = _optimize_with_lbfgs(objective, x0, lo, hi, deadline - 5, max_iter=500)
    else:
        x_opt, f_opt = x0, objective(x0)
    
    return x_opt

def _generate_initial_guesses(problem, wls, n_mat, max_layer, max_total, max_layers):
    """
    Generate diverse initial guesses for optimization.
    """
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    
    candidates = []
    
    # Quarter-wave stacks with different layer counts
    for n_layers in [4, 6, 8, 10, 12, 14, 16, 18, 20]:
        if n_layers > max_layers:
            continue
        for start_char in ['L', 'H']:
            mats = []
            for i in range(n_layers):
                if (i % 2 == 0) == (start_char == 'L'):
                    mats.append('L')
                else:
                    mats.append('H')
            
            # Initial guess: quarter-wave at center wavelength
            wl_center = np.mean(wls)
            thicknesses = []
            for m in mats:
                n = n_L if m == 'L' else n_H
                d = wl_center / (4.0 * n)
                thicknesses.append(d)
            
            candidates.append((mats, thicknesses))
    
    # Multi-quarter-wave stacks with different center wavelengths
    center_wls_options = [
        [500, 600, 700],
        [450, 550, 650],
        [500, 550, 600],
        [400, 600, 800],
    ]
    for n_layers in [6, 8, 10, 12]:
        if n_layers > max_layers:
            continue
        for start_char in ['L', 'H']:
            for cw in center_wls_options:
                mats = []
                for i in range(n_layers):
                    if (i % 2 == 0) == (start_char == 'L'):
                        mats.append('L')
                    else:
                        mats.append('H')
                
                # Assign different center wavelengths to different layers
                thicknesses = []
                for i, m in enumerate(mats):
                    n = n_L if m == 'L' else n_H
                    wl = cw[i % len(cw)]
                    d = wl / (4.0 * n)
                    thicknesses.append(d)
                
                candidates.append((mats, thicknesses))
    
    # Random stacks
    for _ in range(10):
        n_layers = random.randint(6, min(20, max_layers))
        mats = []
        for _ in range(n_layers):
            mats.append(random.choice(['L', 'H']))
        
        # Random thicknesses
        thicknesses = np.random.uniform(0, max_layer, n_layers)
        total = np.sum(thicknesses)
        if total > max_total:
            thicknesses = thicknesses * (max_total / total)
        
        candidates.append((mats, list(thicknesses)))
    
    # Gradient-based initial guess: smooth transition
    n_layers = 12
    positions = np.linspace(0.01, 0.99, n_layers)
    # Smooth transition from ambient to substrate
    n_values = n_ambient + (n_substrate - n_ambient) * (positions ** 1.5)
    
    mats = []
    thicknesses = []
    total_thickness = min(max_total, n_layers * max_layer * 0.8)
    layer_thick = total_thickness / n_layers
    
    for i in range(n_layers):
        n_val = n_values[i]
        if abs(n_val - n_L) < abs(n_val - n_H):
            mats.append('L')
        else:
            mats.append('H')
        thicknesses.append(layer_thick)
    
    candidates.append((mats, thicknesses))
    
    return candidates

def design(problem):
    """
    Design a broadband antireflection coating using L-BFGS-B optimization
    with diverse initial guesses.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

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

    def update_best(mats, ths):
        nonlocal best_val, best_mats, best_ths
        ths_arr = np.array(ths, dtype=float)
        ths_arr = _project_to_feasible(ths_arr, max_layer, max_total)
        
        # Remove zero layers to reduce layer count
        final_mats = []
        final_ths = []
        for m, t in zip(mats, ths_arr):
            if t > 1e-6:
                final_mats.append(m)
                final_ths.append(float(t))
        
        if len(final_mats) > max_layers:
            return
        
        val = _mean_reflectance_percent(final_mats, final_ths, problem, wls, n_mat)
        if val < best_val:
            best_val = val
            best_mats = list(final_mats)
            best_ths = list(final_ths)
        return val

    # Generate diverse initial guesses
    candidates = _generate_initial_guesses(problem, wls, n_mat, max_layer, max_total, max_layers)
    
    # Shuffle candidates
    random.shuffle(candidates)
    
    # Optimize each candidate
    for mats, ths in candidates:
        if time.time() > deadline - 10:
            break
        
        x_opt = _optimize_stack(mats, ths, problem, wls, n_mat, deadline - 5, max_layer, max_total)
        
        # Update best
        update_best(mats, x_opt)
        
        # Also try with the original thicknesses in case optimization failed
        update_best(mats, ths)
    
    # Fallback
    if best_mats is None:
        best_mats = ['L']
        best_ths = [600.0 / (4.0 * n_L)]
        update_best(best_mats, best_ths)
    
    # Final constraint check
    best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
    s = sum(best_ths)
    if s > max_total + 1e-6:
        f = max_total / s
        best_ths = [t * f for t in best_ths]
        best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
    
    return list(best_mats), list(best_ths)
