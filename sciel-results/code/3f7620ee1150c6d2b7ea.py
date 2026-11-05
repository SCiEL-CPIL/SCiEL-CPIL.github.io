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

def _nelder_mead(objective, x0, bounds, deadline, max_iter=300, tol=1e-7):
    """
    Nelder-Mead optimization with bounds handling and adaptive step sizing.
    x0: initial point
    bounds: tuple (lo, hi) arrays
    """
    x0 = np.array(x0, dtype=float)
    n = len(x0)
    lo, hi = bounds
    
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        range_i = hi[i] - lo[i]
        step = 0.1 * range_i if range_i > 1e-6 else 1.0
        xi[i] += step
        simplex.append(np.clip(xi, lo, hi))
        
    f_vals = [objective(x) for x in simplex]
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    best_f = min(f_vals)
    stagnant_count = 0
    
    for iter_count in range(max_iter):
        if time.time() > deadline:
            break
            
        order = np.argsort(f_vals)
        simplex = [simplex[i] for i in order]
        f_vals = [f_vals[i] for i in order]
        
        spread = abs(f_vals[0] - f_vals[-1])
        if spread < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        x_r = np.clip(centroid + alpha * (centroid - simplex[-1]), lo, hi)
        f_r = objective(x_r)
        
        if f_vals[0] <= f_r < f_vals[-2]:
            simplex[-1] = x_r
            f_vals[-1] = f_r
            stagnant_count = 0
            continue
            
        if f_r < f_vals[0]:
            x_e = np.clip(centroid + gamma * (x_r - centroid), lo, hi)
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1] = f_e
            else:
                simplex[-1] = x_r
                f_vals[-1] = f_r
            stagnant_count = 0
            continue
            
        x_c = np.clip(centroid + rho * (simplex[-1] - centroid), lo, hi)
        f_c = objective(x_c)
        if f_c < f_vals[-1]:
            simplex[-1] = x_c
            f_vals[-1] = f_c
            stagnant_count = 0
            continue
            
        for i in range(1, n + 1):
            simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), lo, hi)
            f_vals[i] = objective(simplex[i])
            
        if min(f_vals) > best_f - 1e-10:
            stagnant_count += 1
        else:
            stagnant_count = 0
            best_f = min(f_vals)
            
        if stagnant_count > 10:
            x_best = simplex[0].copy()
            simplex = [x_best.copy()]
            for i in range(n):
                xi = x_best.copy()
                range_i = hi[i] - lo[i]
                step = 0.3 * range_i if range_i > 1e-6 else 3.0
                xi[i] += step
                simplex.append(np.clip(xi, lo, hi))
            f_vals = [objective(x) for x in simplex]
            stagnant_count = 0
            
    order = np.argsort(f_vals)
    best_idx = order[0]
    return np.clip(simplex[best_idx], lo, hi), f_vals[best_idx]

def _coordinate_descent(objective, x0, bounds, deadline, max_iter=500, tol=1e-8):
    """
    Coordinate descent with adaptive step sizes for final polish.
    """
    x = np.array(x0, dtype=float)
    n = len(x)
    lo, hi = bounds
    
    # Initial step sizes
    steps = np.zeros(n)
    for i in range(n):
        range_i = hi[i] - lo[i]
        steps[i] = 0.05 * range_i if range_i > 1e-6 else 0.1
    
    f_best = objective(x)
    best_x = x.copy()
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
            
        improved = False
        for i in range(n):
            # Try positive direction
            x_trial = x.copy()
            x_trial[i] = np.clip(x[i] + steps[i], lo[i], hi[i])
            if not np.array_equal(x_trial, x):
                f_trial = objective(x_trial)
                if f_trial < f_best:
                    x = x_trial
                    f_best = f_trial
                    best_x = x.copy()
                    improved = True
                else:
                    steps[i] *= 0.5
            else:
                steps[i] *= 0.5
            
            # Try negative direction
            x_trial = x.copy()
            x_trial[i] = np.clip(x[i] - steps[i], lo[i], hi[i])
            if not np.array_equal(x_trial, x):
                f_trial = objective(x_trial)
                if f_trial < f_best:
                    x = x_trial
                    f_best = f_trial
                    best_x = x.copy()
                    improved = True
                else:
                    steps[i] *= 0.5
            else:
                steps[i] *= 0.5
        
        if not improved:
            # All steps too small or no improvement, try smaller steps
            min_step = np.min(steps[steps > 0]) if np.any(steps > 0) else 0
            if min_step < tol:
                break
            steps *= 0.5
    
    return best_x, f_best

def _project_to_constraints(th, max_layer, max_total):
    """Project thicknesses onto the constraint box [0, max_layer] and sum <= max_total."""
    th = np.clip(th, 0.0, max_layer)
    s = np.sum(th)
    if s > max_total:
        th = th * (max_total / s)
        th = np.clip(th, 0.0, max_layer)
    return th

def design(problem):
    """
    Design a broadband antireflection coating.
    """
    start_time = time.time()
    deadline = start_time + 140.0

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
    
    sequences = [seq for seq in sequences if len(seq) <= max_layers]
    
    # Generate multiple initial guesses based on quarter-wave stacks
    center_wls = [450, 500, 550, 600, 650, 700, 525, 575, 625, 675]
    
    for mats in sequences:
        if time.time() > deadline:
            break
            
        n_layers = len(mats)
        lo = np.zeros(n_layers)
        hi = np.full(n_layers, max_layer)
        
        initial_guesses = []
        
        for cw in center_wls:
            ths = []
            for m in mats:
                n = n_mat[m]
                q = cw / (4.0 * n)
                q = q * random.uniform(0.9, 1.1)
                ths.append(q)
            initial_guesses.append(ths)
            
        for _ in range(4):
            ths = []
            for _ in range(n_layers):
                if random.random() < 0.5:
                    ths.append(random.uniform(10, 100))
                else:
                    ths.append(random.uniform(50, 250))
            initial_guesses.append(ths)
        
        for x0 in initial_guesses:
            if time.time() > deadline:
                break
                
            x0 = _project_to_constraints(np.array(x0), max_layer, max_total)
            
            def objective(th, m=mats):
                return _mean_reflectance_percent(m, list(th), problem, wls, n_mat)
            
            # Use fewer iterations for longer stacks to fit within time budget
            max_iter = 150 if len(mats) <= 8 else 100
            x_opt, f_opt = _nelder_mead(objective, x0, (lo, hi), deadline, max_iter=max_iter)
            
            x_opt = _project_to_constraints(x_opt, max_layer, max_total)
            val = _mean_reflectance_percent(mats, list(x_opt), problem, wls, n_mat)
            
            if val < best_val:
                best_val = val
                best_mats = mats
                best_ths = list(x_opt)
                
    if best_mats is not None and time.time() < deadline - 10.0:
        n_layers = len(best_mats)
        lo = np.zeros(n_layers)
        hi = np.full(n_layers, max_layer)
        
        def objective(th, m=best_mats):
            return _mean_reflectance_percent(m, list(th), problem, wls, n_mat)
        
        # First, Nelder-Mead polish
        x_polish, f_polish = _nelder_mead(objective, np.array(best_ths), (lo, hi), 
                                          deadline, max_iter=300, tol=1e-9)
        
        x_polish = _project_to_constraints(x_polish, max_layer, max_total)
        val_polish = _mean_reflectance_percent(best_mats, list(x_polish), problem, wls, n_mat)
        
        if val_polish < best_val:
            best_val = val_polish
            best_ths = list(x_polish)
        
        # Then, coordinate descent for finer polish if time remains
        if time.time() < deadline - 5.0:
            x_cd, f_cd = _coordinate_descent(objective, np.array(best_ths), (lo, hi), 
                                             deadline, max_iter=1000, tol=1e-10)
            
            x_cd = _project_to_constraints(x_cd, max_layer, max_total)
            val_cd = _mean_reflectance_percent(best_mats, list(x_cd), problem, wls, n_mat)
            
            if val_cd < best_val:
                best_val = val_cd
                best_ths = list(x_cd)
            
    if best_mats is None:
        best_mats = ['L']
        best_ths = [600.0 / (4.0 * n_L)]
        best_val = _mean_reflectance_percent(best_mats, best_ths, problem, wls, n_mat)
        
    best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
    if sum(best_ths) > max_total + 1e-6:
        f = max_total / sum(best_ths)
        best_ths = [t * f for t in best_ths]
        
    return list(best_mats), list(best_ths)
