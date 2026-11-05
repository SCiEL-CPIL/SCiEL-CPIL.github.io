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

def _nelder_mead(objective, x0, bounds, deadline, max_iter=300, tol=1e-8):
    """
    Nelder-Mead optimization with bounds handling and adaptive step sizing.
    """
    x0 = np.array(x0, dtype=float)
    n = len(x0)
    lo, hi = bounds
    
    # Initialize simplex
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        range_i = hi[i] - lo[i]
        # Smaller initial step for better local convergence
        step = 0.05 * range_i if range_i > 1e-6 else 0.1
        xi[i] += step
        simplex.append(np.clip(xi, lo, hi))
        
    f_vals = [objective(x) for x in simplex]
    best_f = min(f_vals)
    stagnant_count = 0
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for iter_count in range(max_iter):
        if time.time() > deadline:
            break
            
        # Sort simplex
        order = np.argsort(f_vals)
        simplex = [simplex[i] for i in order]
        f_vals = [f_vals[i] for i in order]
        
        # Check convergence
        if abs(f_vals[0] - f_vals[-1]) < tol:
            break
            
        # Centroid of best n points
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = np.clip(centroid + alpha * (centroid - simplex[-1]), lo, hi)
        f_r = objective(x_r)
        
        if f_vals[0] <= f_r < f_vals[-2]:
            simplex[-1] = x_r
            f_vals[-1] = f_r
            if f_r < best_f:
                best_f = f_r
                stagnant_count = 0
            else:
                stagnant_count += 1
            continue
            
        # Expansion
        if f_r < f_vals[0]:
            x_e = np.clip(centroid + gamma * (x_r - centroid), lo, hi)
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1] = f_e
                if f_e < best_f:
                    best_f = f_e
                    stagnant_count = 0
                else:
                    stagnant_count += 1
            else:
                simplex[-1] = x_r
                f_vals[-1] = f_r
                if f_r < best_f:
                    best_f = f_r
                    stagnant_count = 0
                else:
                    stagnant_count += 1
            continue
            
        # Contraction
        x_c = np.clip(centroid + rho * (simplex[-1] - centroid), lo, hi)
        f_c = objective(x_c)
        if f_c < f_vals[-1]:
            simplex[-1] = x_c
            f_vals[-1] = f_c
            if f_c < best_f:
                best_f = f_c
                stagnant_count = 0
            else:
                stagnant_count += 1
            continue
            
        # Shrink
        for i in range(1, n + 1):
            simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), lo, hi)
            f_vals[i] = objective(simplex[i])
        if min(f_vals[1:]) < best_f:
            best_f = min(f_vals[1:])
            stagnant_count = 0
        else:
            stagnant_count += 1
            
        # If stagnant, shrink the whole simplex more aggressively
        if stagnant_count > 10:
            for i in range(1, n + 1):
                simplex[i] = np.clip(simplex[0] + 0.25 * (simplex[i] - simplex[0]), lo, hi)
                f_vals[i] = objective(simplex[i])
            stagnant_count = 0

    order = np.argsort(f_vals)
    best_idx = order[0]
    return np.clip(simplex[best_idx], lo, hi), f_vals[best_idx]

def design(problem):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
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
    
    # Generate candidate sequences
    sequences = []
    
    # Alternating sequences
    for length in [2, 4, 6, 8, 10, 12, 14]:
        if length <= max_layers:
            seq_lh = ['L' if i % 2 == 0 else 'H' for i in range(length)]
            seq_hl = ['H' if i % 2 == 0 else 'L' for i in range(length)]
            sequences.append(seq_lh)
            sequences.append(seq_hl)
            
    # Some non-alternating patterns
    sequences.append(['L', 'L', 'H', 'H'])
    sequences.append(['H', 'H', 'L', 'L'])
    sequences.append(['L', 'H', 'H', 'L'])
    sequences.append(['H', 'L', 'L', 'H'])
    sequences.append(['L', 'L', 'H', 'H', 'L', 'L', 'H', 'H'])
    sequences.append(['H', 'H', 'L', 'L', 'H', 'H', 'L', 'L'])
    
    # Filter by max layers
    sequences = [seq for seq in sequences if len(seq) <= max_layers]
    
    # Center wavelengths for quarter-wave initialization
    center_wls = [450, 500, 550, 600, 650, 700]
    
    for mats in sequences:
        if time.time() > deadline:
            break
            
        n_layers = len(mats)
        lo = np.zeros(n_layers)
        hi = np.full(n_layers, max_layer)
        
        # Generate initial guesses
        initial_guesses = []
        
        # Quarter-wave guesses
        for cw in center_wls:
            ths = []
            for m in mats:
                n = n_mat[m]
                q = cw / (4.0 * n)
                ths.append(q)
            initial_guesses.append(ths)
            
        # Random guesses
        for _ in range(3):
            ths = [random.uniform(10, 200) for _ in range(n_layers)]
            initial_guesses.append(ths)
            
        # Effective medium guess: distribute total thickness to match effective index
        # Target effective index ~ sqrt(1.0 * 1.52) = 1.23
        # This is a rough heuristic
        total_th = max_total * 0.8
        avg_th = total_th / n_layers
        initial_guesses.append([avg_th] * n_layers)
        
        for x0 in initial_guesses:
            if time.time() > deadline:
                break
                
            x0 = np.clip(x0, lo, hi)
            s = sum(x0)
            if s > max_total:
                x0 = x0 * (max_total / s)
                x0 = np.clip(x0, 0.0, max_layer)
            
            def objective(th, m=mats):
                return _mean_reflectance_percent(m, list(th), problem, wls, n_mat)
            
            # Run Nelder-Mead
            x_opt, f_opt = _nelder_mead(objective, x0, (lo, hi), deadline, max_iter=250)
            
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
