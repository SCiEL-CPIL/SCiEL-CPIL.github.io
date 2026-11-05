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

    # Vectorized beta computation
    ns = np.array([n_mat[m] for m in materials])
    ds = np.array(thicknesses, dtype=float)
    beta = 2.0 * math.pi * np.outer(ns * ds, 1.0 / wls)

    cosb = np.cos(beta)
    sinb = np.sin(beta)

    # Build the total characteristic matrix
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

def _nelder_mead(objective, x0, lo, hi, deadline, max_iter=300, tol=1e-10):
    """
    Nelder-Mead with adaptive step size and stagnation detection.
    """
    x0 = np.array(x0, dtype=float)
    n = len(x0)
    
    if n == 0:
        return x0, objective(x0)

    max_layer = hi[0]
    max_total = np.sum(hi)

    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        step = 0.1 * (hi[i] - lo[i]) if hi[i] - lo[i] > 1e-6 else 1.0
        xi[i] += step
        xi = _project_to_feasible(xi, max_layer, max_total)
        simplex.append(xi)

    f_vals = [objective(x) for x in simplex]
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    stagnation_count = 0
    best_f_so_far = min(f_vals)
    stagnation_window = 20
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break

        order = np.argsort(f_vals)
        simplex = [simplex[i] for i in order]
        f_vals = [f_vals[i] for i in order]

        if abs(f_vals[0] - f_vals[-1]) < tol:
            break

        if f_vals[0] < best_f_so_far:
            best_f_so_far = f_vals[0]
            stagnation_count = 0
        else:
            stagnation_count += 1
        
        if stagnation_count >= stagnation_window:
            # Reset with larger steps to escape local minima
            for i in range(1, n + 1):
                simplex[i] = _project_to_feasible(simplex[0] + 2.0 * (simplex[i] - simplex[0]), max_layer, max_total)
                f_vals[i] = objective(simplex[i])
            stagnation_count = 0
            continue

        centroid = np.mean(simplex[:-1], axis=0)

        x_r = _project_to_feasible(centroid + alpha * (centroid - simplex[-1]), max_layer, max_total)
        f_r = objective(x_r)

        if f_vals[0] <= f_r < f_vals[-2]:
            simplex[-1] = x_r
            f_vals[-1] = f_r
            continue

        if f_r < f_vals[0]:
            x_e = _project_to_feasible(centroid + gamma * (x_r - centroid), max_layer, max_total)
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1] = f_e
            else:
                simplex[-1] = x_r
                f_vals[-1] = f_r
            continue

        x_c = _project_to_feasible(centroid + rho * (simplex[-1] - centroid), max_layer, max_total)
        f_c = objective(x_c)
        if f_c < f_vals[-1]:
            simplex[-1] = x_c
            f_vals[-1] = f_c
            continue

        for i in range(1, n + 1):
            simplex[i] = _project_to_feasible(simplex[0] + sigma * (simplex[i] - simplex[0]), max_layer, max_total)
            f_vals[i] = objective(simplex[i])

    order = np.argsort(f_vals)
    best_idx = order[0]
    return simplex[best_idx], f_vals[best_idx]

def _coordinate_descent(objective, x0, lo, hi, deadline, max_iter=200):
    """
    Coordinate descent with adaptive step size.
    """
    x = np.array(x0, dtype=float)
    n = len(x)
    f = objective(x)
    
    max_layer = hi[0]
    max_total = np.sum(hi)
    
    step_sizes = np.array([(hi[i] - lo[i]) * 0.1 if hi[i] - lo[i] > 1e-6 else 1.0 for i in range(n)])
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            for direction in [-1, 1]:
                x_trial = x.copy()
                x_trial[i] += direction * step_sizes[i]
                x_trial = _project_to_feasible(x_trial, max_layer, max_total)
                f_trial = objective(x_trial)
                
                if f_trial < f:
                    x = x_trial
                    f = f_trial
                    improved = True
                    step_sizes[i] *= 1.5
                else:
                    step_sizes[i] *= 0.5
        
        if not improved:
            break
    
    return x, f

def _quarter_wave_stack(n_layers, start_char, n_L, n_H, n_ambient, n_substrate, wls):
    """
    Generate a quarter-wave stack with optimized thicknesses.
    """
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
    
    return mats, thicknesses

def _multi_quarter_wave_stack(n_layers, start_char, n_L, n_H, n_ambient, n_substrate, wls, center_wls):
    """
    Generate a multi-quarter-wave stack with different center wavelengths for different layers.
    """
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
        # Cycle through center wavelengths
        wl = center_wls[i % len(center_wls)]
        d = wl / (4.0 * n)
        thicknesses.append(d)
    
    return mats, thicknesses

def _random_stack(n_layers, max_layer, max_total):
    """
    Generate a random stack.
    """
    mats = []
    for _ in range(n_layers):
        mats.append(random.choice(['L', 'H']))
    
    # Random thicknesses
    thicknesses = [random.uniform(10, max_layer) for _ in range(n_layers)]
    # Scale to fit total constraint if needed
    s = sum(thicknesses)
    if s > max_total:
        f = max_total / s
        thicknesses = [t * f for t in thicknesses]
    
    return mats, thicknesses

def design(problem):
    """
    Design a broadband antireflection coating using a search-based approach.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

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
        ths_clipped = [min(max(t, 0.0), max_layer) for t in ths]
        s = sum(ths_clipped)
        if s > max_total:
            if s > 1e-9:
                f = max_total / s
                ths_clipped = [t * f for t in ths_clipped]
        val = _mean_reflectance_percent(mats, ths_clipped, problem, wls, n_mat)
        if val < best_val:
            best_val = val
            best_mats = list(mats)
            best_ths = list(ths_clipped)
        return val

    # Generate diverse initial designs and optimize them
    # Strategy: for each layer count, try multiple starting points
    
    # Common quarter-wave designs
    center_wls = [450, 500, 550, 600, 650, 700]
    
    # Build list of (materials, initial_thicknesses) candidates
    candidates = []
    
    for n_layers in [2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20]:
        if n_layers > max_layers:
            continue
        # Alternating starting with L or H
        for start_char in ['L', 'H']:
            mats_qw, ths_qw = _quarter_wave_stack(n_layers, start_char, n_L, n_H, n_ambient, n_substrate, wls)
            candidates.append((mats_qw, ths_qw))
            
            mats_mqw, ths_mqw = _multi_quarter_wave_stack(n_layers, start_char, n_L, n_H, n_ambient, n_substrate, wls, center_wls)
            candidates.append((mats_mqw, ths_mqw))
            
            # Add random initial points
            for _ in range(2):
                mats_r, ths_r = _random_stack(n_layers, max_layer, max_total)
                candidates.append((mats_r, ths_r))

    # Shuffle for diversity
    random.shuffle(candidates)
    
    # Limit number of candidates based on time
    max_candidates = 80
    candidates = candidates[:max_candidates]
    
    for mats, ths_init in candidates:
        if time.time() > deadline:
            break
        
        n_layers = len(mats)
        lo = np.zeros(n_layers)
        hi = np.full(n_layers, max_layer)
        
        x0 = np.clip(np.array(ths_init, dtype=float), lo, hi)
        
        def objective(th, m=mats):
            return _mean_reflectance_percent(m, list(th), problem, wls, n_mat)
        
        # First do coordinate descent for quick improvement
        x_cd, f_cd = _coordinate_descent(objective, x0, lo, hi, deadline, max_iter=50)
        
        # Then Nelder-Mead for refinement
        x_nm, f_nm = _nelder_mead(objective, x_cd, lo, hi, deadline, max_iter=200)
        
        # Use the better of the two
        x_best = x_nm if f_nm <= f_cd else x_cd
        f_best = min(f_nm, f_cd)
        
        update_best(mats, x_best)
        
        # Also try the CD result separately in case NM got stuck
        if f_cd < f_nm:
            update_best(mats, x_cd)
    
    # Additional targeted search: refine the best found so far
    if best_mats is not None and time.time() < deadline - 10:
        n_layers = len(best_mats)
        lo = np.zeros(n_layers)
        hi = np.full(n_layers, max_layer)
        
        def objective(th):
            return _mean_reflectance_percent(best_mats, list(th), problem, wls, n_mat)
        
        # Multiple restarts from perturbed best
        for _ in range(5):
            if time.time() > deadline:
                break
            x0 = np.array(best_ths, dtype=float).copy()
            # Perturb
            for i in range(n_layers):
                if random.random() < 0.5:
                    x0[i] += random.uniform(-0.1, 0.1) * (hi[i] - lo[i])
                    x0[i] = np.clip(x0[i], lo[i], hi[i])
            
            x_opt, f_opt = _nelder_mead(objective, x0, lo, hi, deadline, max_iter=150)
            update_best(best_mats, x_opt)
    
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
