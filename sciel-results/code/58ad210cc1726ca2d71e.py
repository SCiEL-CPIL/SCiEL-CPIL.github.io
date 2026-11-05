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

def _nelder_mead_fast(objective, x0, lo, hi, deadline, max_iter=200, tol=1e-8):
    """
    Fast Nelder-Mead with bound handling and deadline check.
    """
    x0 = np.array(x0, dtype=float)
    n = len(x0)
    
    if n == 0:
        return x0, objective(x0)

    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        step = 0.1 * (hi[i] - lo[i]) if hi[i] - lo[i] > 1e-6 else 1.0
        xi[i] += step
        xi = _project_to_feasible(xi, hi[0], np.sum(hi))
        simplex.append(xi)

    f_vals = [objective(x) for x in simplex]
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break

        order = np.argsort(f_vals)
        simplex = [simplex[i] for i in order]
        f_vals = [f_vals[i] for i in order]

        if abs(f_vals[0] - f_vals[-1]) < tol:
            break

        centroid = np.mean(simplex[:-1], axis=0)

        x_r = _project_to_feasible(centroid + alpha * (centroid - simplex[-1]), hi[0], np.sum(hi))
        f_r = objective(x_r)

        if f_vals[0] <= f_r < f_vals[-2]:
            simplex[-1] = x_r
            f_vals[-1] = f_r
            continue

        if f_r < f_vals[0]:
            x_e = _project_to_feasible(centroid + gamma * (x_r - centroid), hi[0], np.sum(hi))
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1] = f_e
            else:
                simplex[-1] = x_r
                f_vals[-1] = f_r
            continue

        x_c = _project_to_feasible(centroid + rho * (simplex[-1] - centroid), hi[0], np.sum(hi))
        f_c = objective(x_c)
        if f_c < f_vals[-1]:
            simplex[-1] = x_c
            f_vals[-1] = f_c
            continue

        for i in range(1, n + 1):
            simplex[i] = _project_to_feasible(simplex[0] + sigma * (simplex[i] - simplex[0]), hi[0], np.sum(hi))
            f_vals[i] = objective(simplex[i])

    order = np.argsort(f_vals)
    best_idx = order[0]
    return simplex[best_idx], f_vals[best_idx]

def _coordinate_descent(objective, x0, lo, hi, deadline, max_iter=100):
    """
    Coordinate descent with line search.
    """
    x = np.array(x0, dtype=float)
    n = len(x)
    f = objective(x)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            
            x_old = x.copy()
            f_old = f
            
            # Try moving in positive direction
            step = (hi[i] - lo[i]) * 0.1
            for _ls in range(20):
                x[i] = min(x_old[i] + step, hi[i])
                x = _project_to_feasible(x, hi[0], np.sum(hi))
                f_new = objective(x)
                
                if f_new < f:
                    f = f_new
                    improved = True
                    step *= 0.5
                else:
                    break
            
            # Try moving in negative direction
            x = x_old.copy()
            step = (hi[i] - lo[i]) * 0.1
            for _ls in range(20):
                x[i] = max(x_old[i] - step, lo[i])
                x = _project_to_feasible(x, hi[0], np.sum(hi))
                f_new = objective(x)
                
                if f_new < f:
                    f = f_new
                    improved = True
                    step *= 0.5
                else:
                    break
        
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

def _random_stack(n_layers, max_layer, max_total):
    """
    Generate a random stack.
    """
    mats = []
    for _ in range(n_layers):
        mats.append(random.choice(['L', 'H']))
    
    # Random thicknesses
    thicknesses = np.random.uniform(0, max_layer, n_layers)
    total = np.sum(thicknesses)
    if total > max_total:
        thicknesses = thicknesses * (max_total / total)
    
    return mats, thicknesses

def design(problem):
    """
    Design a broadband antireflection coating using a combination of
    deterministic initialization, coordinate descent, and Nelder-Mead.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

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

    # Generate candidate stacks
    candidates = []
    
    # Quarter-wave stacks with different layer counts
    for n_layers in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
        if n_layers > max_layers:
            continue
        for start_char in ['L', 'H']:
            mats, ths = _quarter_wave_stack(n_layers, start_char, n_L, n_H, n_ambient, n_substrate, wls)
            candidates.append((mats, ths))
    
    # Random stacks
    for _ in range(20):
        n_layers = random.randint(4, min(20, max_layers))
        mats, ths = _random_stack(n_layers, max_layer, max_total)
        candidates.append((mats, list(ths)))
    
    # Shuffle candidates
    random.shuffle(candidates)
    
    # Optimize each candidate
    for mats, ths in candidates:
        if time.time() > deadline - 15:
            break
        
        # First, do coordinate descent
        lo = np.zeros(len(ths))
        hi = np.full(len(ths), max_layer)
        
        def objective(th):
            th = _project_to_feasible(th, max_layer, max_total)
            return _mean_reflectance_percent(mats, list(th), problem, wls, n_mat)
        
        x0 = np.array(ths, dtype=float)
        x_cd, f_cd = _coordinate_descent(objective, x0, lo, hi, deadline - 5, max_iter=50)
        
        # Then, do Nelder-Mead refinement
        if time.time() < deadline - 5:
            x_nm, f_nm = _nelder_mead_fast(objective, x_cd, lo, hi, deadline - 2, max_iter=100)
            if f_nm < f_cd:
                x_cd = x_cd
                x_cd = x_nm
        
        # Update best
        update_best(mats, x_cd)
        
        # Also try with the original thicknesses in case optimization failed
        update_best(mats, x0)
    
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
