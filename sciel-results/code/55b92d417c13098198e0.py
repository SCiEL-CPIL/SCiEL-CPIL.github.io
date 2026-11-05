import time
import math
import numpy as np
import random

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

def _project_constraints(th, max_total, max_layer):
    """
    Project thicknesses onto the feasible region:
    0 <= th_i <= max_layer
    sum(th) <= max_total
    """
    th = np.clip(th, 0.0, max_layer)
    total_th = np.sum(th)
    if total_th > max_total:
        # Scale down proportionally
        if total_th > 0:
            th = th * (max_total / total_th)
        else:
            th = np.zeros_like(th)
        # Re-clip after scaling to ensure box constraints (scaling might push some to 0, but won't exceed max_layer if they were <= max_layer before)
        th = np.clip(th, 0.0, max_layer)
        # It is possible that scaling causes sum < max_total, which is fine.
        # It is also possible that clipping after scaling is needed if some elements were exactly at max_layer and scaled down, but wait:
        # If th_i <= max_layer, then th_i * (max_total/total_th) <= max_layer * (max_total/total_th).
        # Since total_th >= max_layer (if any element is max_layer), the factor is <= 1.
        # So scaled values are <= max_layer.
        # However, if total_th < max_layer, factor > 1? No, total_th is sum of n elements.
        # Actually, if total_th > max_total, then max_total < total_th.
        # The factor is max_total / total_th < 1.
        # So th_i decreases. It cannot exceed max_layer.
        # So re-clipping is redundant but safe.
    return th

def _local_optimize(th_init, mats, problem, wls, n_mat, max_total, max_layer, time_limit=5.0):
    """
    A simple gradient-based local optimizer with projection.
    Uses finite differences for gradients.
    """
    start_time = time.time()
    th = np.array(th_init, dtype=float)
    th = _project_constraints(th, max_total, max_layer)
    
    current_val = _mean_reflectance_percent(mats, list(th), problem, wls, n_mat)
    
    # Finite difference step size
    h = 1e-4
    
    # Simple gradient descent with adaptive step size
    step_size = 1.0
    max_iter = 2000
    
    for i in range(max_iter):
        if time.time() > start_time + time_limit:
            break
            
        # Compute gradient via central differences
        grad = np.zeros_like(th)
        for j in range(len(th)):
            th_plus = th.copy()
            th_plus[j] += h
            th_plus = _project_constraints(th_plus, max_total, max_layer)
            val_plus = _mean_reflectance_percent(mats, list(th_plus), problem, wls, n_mat)
            
            th_minus = th.copy()
            th_minus[j] -= h
            th_minus = _project_constraints(th_minus, max_total, max_layer)
            val_minus = _mean_reflectance_percent(mats, list(th_minus), problem, wls, n_mat)
            
            # Note: If projection changes the point significantly, this gradient is an approximation.
            # For better accuracy, we should compute gradient on the unconstrained space and then project,
            # but that's complex. Here we use the projected points.
            grad[j] = (val_plus - val_minus) / (2.0 * h)
            
        # Check if gradient is small
        if np.max(np.abs(grad)) < 1e-6:
            break
            
        # Update
        th_new = th - step_size * grad
        
        # Project
        th_new = _project_constraints(th_new, max_total, max_layer)
        
        # Line search (simple backtracking)
        val_new = _mean_reflectance_percent(mats, list(th_new), problem, wls, n_mat)
        
        # If value didn't improve, reduce step size
        if val_new > current_val:
            step_size *= 0.5
            if step_size < 1e-6:
                break
            # Try smaller step
            th_new = th - step_size * grad
            th_new = _project_constraints(th_new, max_total, max_layer)
            val_new = _mean_reflectance_percent(mats, list(th_new), problem, wls, n_mat)
            if val_new > current_val:
                break # Stuck
        else:
            # Success, maybe increase step size slightly
            step_size = min(step_size * 1.1, 5.0)
            th = th_new
            current_val = val_new

    return th, current_val

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
    
    # Run local optimization on the top N candidates
    # N is limited by time.
    num_optimizations = 15
    
    for i in range(min(num_optimizations, len(candidates))):
        if time.time() > deadline - 10.0:  # Leave some time for final checks
            break
            
        score_init, mats, x0 = candidates[i]
        
        try:
            x_opt, val_opt = _local_optimize(x0, mats, problem, wls, n_mat, max_total, max_layer, time_limit=5.0)
            
            # Post-process: ensure constraints are strictly met
            x_opt = _project_constraints(x_opt, max_total, max_layer)
            
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
