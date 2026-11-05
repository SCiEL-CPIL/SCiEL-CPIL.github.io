import time
import math
import numpy as np
import random

def _reflectance_array(materials, thicknesses, n_ambient, n_substrate, wls, n_mat):
    """
    Compute reflectance array for given materials and thicknesses using
    the thin-film characteristic-matrix method.
    Vectorized over wavelengths for speed.
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
        if d > 1e-9:
            beta[i, :] = 2.0 * math.pi * n * d / wls

    cosb = np.cos(beta)
    sinb = np.sin(beta)

    # Build the total characteristic matrix
    # M = [[M11, M12], [M21, M22]]
    # Initialize to identity
    M11 = np.ones(n_wls, dtype=complex)
    M12 = np.zeros(n_wls, dtype=complex)
    M21 = np.zeros(n_wls, dtype=complex)
    M22 = np.ones(n_wls, dtype=complex)

    for i, m in enumerate(materials):
        n = n_mat[m]
        c = cosb[i]
        s = sinb[i]
        
        # Matrix elements for layer i
        # M_i = [[c, i*s/n], [i*n*s, c]]
        A = c
        B = 1j * s / n
        C = 1j * n * s
        D = c

        # Matrix multiplication: M_new = M_old * M_i
        # M_old is [[M11, M12], [M21, M22]]
        
        nM11 = M11 * A + M12 * C
        nM12 = M11 * B + M12 * D
        nM21 = M21 * A + M22 * C
        nM22 = M21 * B + M22 * D
        
        M11, M12, M21, M22 = nM11, nM12, nM21, nM22

    # Admittance Y
    # Y = (M21 + n_substrate * M22) / (M11 + n_substrate * M12)
    Y = (M21 + n_substrate * M22) / (M11 + n_substrate * M12)
    
    # Reflection coefficient r
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
    
    # Check constraints
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

def _project_constraints(th, max_total, max_layer):
    """
    Project thicknesses onto the feasible region:
    0 <= th_i <= max_layer
    sum(th) <= max_total
    """
    th = np.clip(th, 0.0, max_layer)
    total_th = np.sum(th)
    if total_th > max_total:
        if total_th > 0:
            th = th * (max_total / total_th)
        else:
            th = np.zeros_like(th)
        # Re-clip after scaling
        th = np.clip(th, 0.0, max_layer)
    return th

def _nelder_mead(objective, x0, bounds, deadline, max_iter=200, tol=1e-6):
    """
    Nelder-Mead optimization with bounds handling.
    x0: initial point
    bounds: tuple (lo, hi) arrays
    """
    x0 = np.array(x0, dtype=float)
    n = len(x0)
    lo, hi = bounds
    
    # Initialize simplex
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        # Use a small step relative to the range or a fixed small step
        range_i = hi[i] - lo[i]
        step = 0.1 * range_i if range_i > 1e-6 else 1.0
        xi[i] += step
        simplex.append(_project_constraints(xi, bounds[1].sum(), hi[0]) if False else np.clip(xi, lo, hi))
        
    # Note: _project_constraints in the call above is a placeholder, we need to pass max_total and max_layer
    # Let's refactor to pass projection params or use a lambda
    
    # Actually, let's just clip for simplex initialization and handle sum constraint in objective?
    # No, better to project properly.
    
    # Let's redefine the projection inside or pass params
    # For simplicity in this function, we assume the objective function handles penalties or we project here.
    # Let's pass max_total and max_layer to the function.
    
    # Re-initialize simplex with proper projection
    # We need max_total and max_layer. Let's assume they are passed or global? 
    # Better to pass them.
    
    # Let's rewrite _nelder_mead to accept projection params
    pass

def _nelder_mead_opt(objective, x0, lo, hi, max_total, max_layer, deadline, max_iter=200, tol=1e-6):
    """
    Nelder-Mead optimization with bounds and sum constraint projection.
    """
    x0 = np.array(x0, dtype=float)
    x0 = _project_constraints(x0, max_total, max_layer)
    n = len(x0)
    
    # Initialize simplex
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        range_i = hi[i] - lo[i]
        step = 0.1 * range_i if range_i > 1e-6 else 1.0
        xi[i] += step
        simplex.append(_project_constraints(xi, max_total, max_layer))
        
    f_vals = [objective(x) for x in simplex]
    
    # Alpha, Gamma, Rho, Sigma
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    best_f = min(f_vals)
    best_x = simplex[f_vals.index(best_f)].copy()
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        # Sort simplex by function value
        order = np.argsort(f_vals)
        simplex = [simplex[idx] for idx in order]
        f_vals = [f_vals[idx] for idx in order]
        
        # Check convergence
        if f_vals[-1] - f_vals[0] < tol:
            break
            
        # Calculate centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = centroid + alpha * (centroid - simplex[-1])
        x_r = _project_constraints(x_r, max_total, max_layer)
        f_r = objective(x_r)
        
        if f_r < f_vals[0]:
            # Expansion
            x_e = centroid + gamma * (x_r - centroid)
            x_e = _project_constraints(x_e, max_total, max_layer)
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1] = f_e
            else:
                simplex[-1] = x_r
                f_vals[-1] = f_r
        elif f_r < f_vals[-2]:
            # Accept reflection
            simplex[-1] = x_r
            f_vals[-1] = f_r
        else:
            # Contraction
            x_c = centroid + 0.5 * (simplex[-1] - centroid)
            x_c = _project_constraints(x_c, max_total, max_layer)
            f_c = objective(x_c)
            if f_c < f_vals[-1]:
                simplex[-1] = x_c
                f_vals[-1] = f_c
            else:
                # Shrink
                for j in range(1, n + 1):
                    simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                    simplex[j] = _project_constraints(simplex[j], max_total, max_layer)
                    f_vals[j] = objective(simplex[j])
        
        # Update best
        current_best_f = min(f_vals)
        if current_best_f < best_f:
            best_f = current_best_f
            best_x = simplex[f_vals.index(current_best_f)].copy()
            
    return best_x, best_f

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
    sequences = []
    
    # Try different numbers of layers
    # Even numbers of layers often allow for symmetric structures which can be broadband
    for n in range(2, min(max_layers + 1, 13), 2):
        # L start
        seq1 = ['L' if i % 2 == 0 else 'H' for i in range(n)]
        sequences.append(seq1)
        # H start
        seq2 = ['H' if i % 2 == 0 else 'L' for i in range(n)]
        sequences.append(seq2)
        
    # Add some specific patterns
    if max_layers >= 4:
        sequences.append(['L', 'H', 'H', 'L'])
        sequences.append(['H', 'L', 'L', 'H'])
    if max_layers >= 6:
        sequences.append(['L', 'H', 'H', 'L', 'H', 'L'])
        sequences.append(['H', 'L', 'L', 'H', 'L', 'H'])
        
    # Filter sequences by max_layers
    sequences = [seq for seq in sequences if len(seq) <= max_layers]
    
    # Generate initial guesses based on quarter-wave stacks
    center_wls = [450, 500, 550, 600, 650, 700]
    
    # We will collect a pool of initial candidates (mats, x0) and their initial scores
    candidates = []
    
    for mats in sequences:
        if time.time() > deadline:
            break
            
        n_layers = len(mats)
        
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
    num_optimizations = 10
    
    lo = np.zeros(max_layers)
    hi = np.full(max_layers, max_layer)
    
    for i in range(min(num_optimizations, len(candidates))):
        if time.time() > deadline - 10.0:  # Leave some time for final checks
            break
            
        score_init, mats, x0 = candidates[i]
        
        try:
            # Define objective function for this specific material stack
            def objective(x):
                x = _project_constraints(x, max_total, max_layer)
                return _mean_reflectance_percent(mats, list(x), problem, wls, n_mat)
            
            x_opt, val_opt = _nelder_mead_opt(objective, x0, lo[:len(x0)], hi[:len(x0)], max_total, max_layer, deadline)
            
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
