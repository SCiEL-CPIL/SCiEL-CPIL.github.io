import time
import math
import numpy as np
import random
import scipy.optimize

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
    # beta shape: (n_layers, n_wls)
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
        # M_old is [[M11, M12], [M21, M22]]
        # M_i is [[A, B], [C, D]]
        
        nM11 = M11 * A + M12 * C
        nM12 = M11 * B + M12 * D
        nM21 = M21 * A + M22 * C
        nM22 = M21 * B + M22 * D
        
        M11, M12, M21, M22 = nM11, nM12, nM21, nM22

    # Admittance Y
    # Y = (M21 + n_substrate * M22) / (M11 + n_substrate * M12)
    Y = (M21 + n_substrate * M22) / (M11 + n_substrate * M12)
    
    # Reflectance r
    # r = (n_ambient - Y) / (n_ambient + Y)
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
        # Penalty proportional to violation
        return 1e6 + (total_th - max_total) * 10.0

    if len(materials) == 0:
        r = (problem["ambient_index"] - problem["substrate_index"]) / (
            problem["ambient_index"] + problem["substrate_index"])
        return r * r * 100.0

    R = _reflectance_array(materials, thicknesses, problem["ambient_index"],
                           problem["substrate_index"], wls, n_mat)
    return float(np.mean(R)) * 100.0

def _objective_function(th, mats, problem, wls, n_mat, max_total, max_layer):
    """
    Objective function for scipy.optimize.minimize.
    Includes penalty for exceeding total thickness.
    """
    # Box constraints are handled by L-BFGS-B, but we can double check or add penalty
    # for total sum to guide optimizer if it drifts towards high sum.
    total_th = np.sum(th)
    penalty = 0.0
    if total_th > max_total:
        penalty = (total_th - max_total) * 10.0
    
    base_val = _mean_reflectance_percent(mats, list(th), problem, wls, n_mat)
    return base_val + penalty

def design(problem):
    """
    Design a broadband antireflection coating.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

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
    # Focus on alternating stacks which are known to be effective for broadband AR
    sequences = []
    
    # Length 2 to 10 (even numbers usually better for symmetry/broadband)
    for n in range(2, 11, 2):
        if n > max_layers:
            break
        # L start
        seq1 = ['L' if i % 2 == 0 else 'H' for i in range(n)]
        sequences.append(seq1)
        # H start
        seq2 = ['H' if i % 2 == 0 else 'L' for i in range(n)]
        sequences.append(seq2)
        
    # Add some specific patterns if space allows
    if max_layers >= 4:
        sequences.append(['L', 'H', 'H', 'L'])
        sequences.append(['H', 'L', 'L', 'H'])
    if max_layers >= 6:
        sequences.append(['L', 'H', 'H', 'L', 'H', 'L'])
        sequences.append(['H', 'L', 'L', 'H', 'L', 'H'])
        
    # Filter sequences by max_layers
    sequences = [seq for seq in sequences if len(seq) <= max_layers]
    
    # Generate initial guesses based on quarter-wave stacks
    # at different center wavelengths to cover the band
    center_wls = [450, 500, 550, 600, 650, 700, 525, 575, 625, 675]
    
    # We will collect a pool of initial candidates (mats, x0) and their initial scores
    candidates = []
    
    for mats in sequences:
        if time.time() > deadline:
            break
            
        n_layers = len(mats)
        lo = np.zeros(n_layers)
        hi = np.full(n_layers, max_layer)
        
        # Generate initial guesses
        for cw in center_wls:
            ths = []
            for m in mats:
                n = n_mat[m]
                q = cw / (4.0 * n)
                ths.append(q)
            
            x0 = np.array(ths)
            # Scale to fit total thickness constraint if needed
            s = sum(x0)
            if s > max_total:
                x0 = x0 * (max_total / s)
                x0 = np.clip(x0, 0.0, max_layer)
            else:
                x0 = np.clip(x0, 0.0, max_layer)
                
            # Calculate initial score to prioritize good starts
            score = _mean_reflectance_percent(mats, list(x0), problem, wls, n_mat)
            candidates.append((score, mats, x0))
            
            # Add a slightly perturbed version to escape symmetry
            x0_pert = x0 * np.array([random.uniform(0.9, 1.1) for _ in range(n_layers)])
            s_pert = sum(x0_pert)
            if s_pert > max_total:
                x0_pert = x0_pert * (max_total / s_pert)
                x0_pert = np.clip(x0_pert, 0.0, max_layer)
            else:
                x0_pert = np.clip(x0_pert, 0.0, max_layer)
                
            score_pert = _mean_reflectance_percent(mats, list(x0_pert), problem, wls, n_mat)
            candidates.append((score_pert, mats, x0_pert))
            
    # Sort candidates by initial score
    candidates.sort(key=lambda x: x[0])
    
    # Run L-BFGS-B on the top N candidates
    # N is limited by time. Each run is fast, but let's say we try top 5-10.
    num_optimizations = 10
    
    for i in range(min(num_optimizations, len(candidates))):
        if time.time() > deadline - 5.0:  # Leave some time for final checks
            break
            
        score_init, mats, x0 = candidates[i]
        
        def objective(th, m=mats):
            return _objective_function(th, m, problem, wls, n_mat, max_total, max_layer)
            
        try:
            # L-BFGS-B is robust for box-constrained problems
            result = scipy.optimize.minimize(objective, x0, method='L-BFGS-B', 
                                             bounds=list(zip(lo, hi)), 
                                             options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            
            x_opt = result.x
            # Post-process: ensure constraints are strictly met
            x_opt = np.clip(x_opt, 0.0, max_layer)
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
            continue

    # If no optimization succeeded, fallback to best initial guess
    if best_mats is None and len(candidates) > 0:
        best_val, best_mats, best_ths = candidates[0][0], candidates[0][1], list(candidates[0][2])
        # Ensure constraints
        best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
        s = sum(best_ths)
        if s > max_total:
            f = max_total / s
            best_ths = [t * f for t in best_ths]
            best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
            
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
        # Re-clip after scaling
        best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
        
    return list(best_mats), list(best_ths)
