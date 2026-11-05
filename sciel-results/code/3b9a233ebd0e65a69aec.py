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

def _nelder_mead(objective, x0, lo, hi, deadline, max_iter=300, tol=1e-10):
    """
    Nelder-Mead with adaptive step size and stagnation detection.
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
            # Reset with larger steps
            for i in range(1, n + 1):
                simplex[i] = _project_to_feasible(simplex[0] + 2.0 * (simplex[i] - simplex[0]), hi[0], np.sum(hi))
                f_vals[i] = objective(simplex[i])
            stagnation_count = 0
            continue

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

def _coordinate_descent(objective, x0, lo, hi, deadline, max_iter=200):
    """
    Coordinate descent with adaptive step size and line search.
    """
    x = np.array(x0, dtype=float)
    n = len(x)
    f = objective(x)
    
    step_sizes = np.array([(hi[i] - lo[i]) * 0.1 if hi[i] - lo[i] > 1e-6 else 1.0 for i in range(n)])
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            # Try multiple step sizes for each coordinate
            best_x_i = x[i]
            best_f_i = f
            best_step = 0.0
            
            for step_mult in [1.0, 0.5, 0.25, 2.0, 4.0]:
                for direction in [-1, 1]:
                    x_trial = x.copy()
                    x_trial[i] += direction * step_mult * step_sizes[i]
                    x_trial = _project_to_feasible(x_trial, hi[0], np.sum(hi))
                    f_trial = objective(x_trial)
                    
                    if f_trial < best_f_i:
                        best_f_i = f_trial
                        best_x_i = x_trial[i]
                        best_step = step_mult * step_sizes[i]
            
            if best_step != 0.0:
                x[i] = best_x_i
                f = best_f_i
                improved = True
                # Adjust step size based on what worked
                if abs(best_step) > step_sizes[i]:
                    step_sizes[i] = abs(best_step) * 0.8
                else:
                    step_sizes[i] = max(step_sizes[i] * 0.7, 1e-4)
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
    thicknesses = np.random.uniform(0, max_layer, n_layers)
    total = np.sum(thicknesses)
    if total > max_total:
        thicknesses = thicknesses * (max_total / total)
    
    return mats, list(thicknesses)

def _optimize_stack(mats, ths, problem, wls, n_mat, deadline, max_layer, max_total):
    """
    Optimize a single stack using coordinate descent followed by Nelder-Mead.
    """
    lo = np.zeros(len(ths))
    hi = np.full(len(ths), max_layer)
    
    def objective(th):
        th = _project_to_feasible(th, max_layer, max_total)
        return _mean_reflectance_percent(mats, list(th), problem, wls, n_mat)
    
    x0 = np.array(ths, dtype=float)
    x0 = _project_to_feasible(x0, max_layer, max_total)
    
    # First, coordinate descent for quick improvement
    if time.time() < deadline - 5:
        x_cd, f_cd = _coordinate_descent(objective, x0, lo, hi, deadline - 5, max_iter=150)
    else:
        x_cd, f_cd = x0, objective(x0)
    
    # Then, Nelder-Mead for local refinement
    if time.time() < deadline - 2:
        x_nm, f_nm = _nelder_mead(objective, x_cd, lo, hi, deadline - 2, max_iter=300, tol=1e-12)
        if f_nm < f_cd:
            x_cd = x_nm
            f_cd = f_nm
    
    # Final high-precision polish pass
    if time.time() < deadline - 1:
        # Use a very tight Nelder-Mead with small initial simplex
        x0_polish = x_cd.copy()
        n = len(x0_polish)
        simplex = [x0_polish.copy()]
        for i in range(n):
            xi = x0_polish.copy()
            xi[i] += 0.01 * (hi[i] - lo[i])
            xi = _project_to_feasible(xi, hi[0], np.sum(hi))
            simplex.append(xi)
        
        f_vals = [objective(x) for x in simplex]
        best_polish_f = min(f_vals)
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        tol = 1e-14
        
        for _ in range(100):
            if time.time() > deadline - 1:
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
        if f_vals[best_idx] < f_cd:
            x_cd = simplex[best_idx]
    
    return x_cd

def _smooth_profile_to_layers(profile, n_L, n_H, n_ambient, n_substrate, max_layer, max_total, n_layers=12):
    """
    Convert a smooth refractive index profile to discrete layers.
    The profile is a function of position in [0, 1].
    We discretize into n_layers and assign L or H based on whether n is closer to n_L or n_H.
    """
    # Discretize the profile into n_layers
    positions = np.linspace(0.01, 0.99, n_layers)
    n_values = profile(positions)
    
    # Assign material based on which is closer
    mats = []
    thicknesses = []
    
    # Calculate layer thicknesses (equal spacing)
    total_thickness = min(max_total, n_layers * max_layer * 0.8)
    layer_thick = total_thickness / n_layers
    
    for i in range(n_layers):
        n_val = n_values[i]
        if abs(n_val - n_L) < abs(n_val - n_H):
            mats.append('L')
        else:
            mats.append('H')
        thicknesses.append(layer_thick)
    
    return mats, thicknesses

def _optimize_smooth_profile(problem, wls, n_mat, deadline, max_layer, max_total):
    """
    Optimize a smooth refractive index profile using Nelder-Mead.
    The profile is defined by a few parameters that create a smooth transition.
    """
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    
    # Use a sigmoid-based profile: n(x) = n_ambient + (n_substrate - n_ambient) * sigmoid(k*(x-c))
    # Parameters: k (steepness), c (center position), n_mid (mid-point adjustment)
    # Actually, let's use a simpler approach: linear interpolation with some flexibility
    # n(x) = n_ambient + (n_substrate - n_ambient) * x^alpha
    # alpha controls the shape: alpha=1 is linear, alpha>1 is convex, alpha<1 is concave
    
    def objective(params):
        alpha = params[0]
        beta = params[1]
        gamma = params[2]
        
        # Create profile: n(x) = n_ambient + (n_substrate - n_ambient) * (x^alpha)
        # But we need to map to L/H layers, so let's just use the profile to guide layer assignment
        # For now, use a simple approach: optimize the alpha parameter for the profile shape
        
        # Generate profile
        def profile(x):
            return n_ambient + (n_substrate - n_ambient) * (x ** alpha)
        
        mats, ths = _smooth_profile_to_layers(profile, n_L, n_H, n_ambient, n_substrate, max_layer, max_total, n_layers=10)
        
        if len(mats) == 0:
            return 1e6
        
        return _mean_reflectance_percent(mats, ths, problem, wls, n_mat)
    
    # Initial guess: alpha=1 (linear)
    x0 = np.array([1.0, 0.5, 0.5])
    lo = np.array([0.1, 0.0, 0.0])
    hi = np.array([5.0, 1.0, 1.0])
    
    x_opt, f_opt = _nelder_mead(objective, x0, lo, hi, deadline, max_iter=100, tol=1e-8)
    
    # Get the best profile
    def profile(x):
        alpha = x_opt[0]
        return n_ambient + (n_substrate - n_ambient) * (x ** alpha)
    
    mats, ths = _smooth_profile_to_layers(profile, n_L, n_H, n_ambient, n_substrate, max_layer, max_total, n_layers=10)
    
    return mats, ths

def design(problem):
    """
    Design a broadband antireflection coating using coordinate descent
    and Nelder-Mead for local refinement with diverse initial guesses.
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
    candidates = []
    
    # Quarter-wave stacks with different layer counts
    for n_layers in [4, 6, 8, 10, 12, 14, 16, 18, 20]:
        if n_layers > max_layers:
            continue
        for start_char in ['L', 'H']:
            mats, ths = _quarter_wave_stack(n_layers, start_char, n_L, n_H, n_ambient, n_substrate, wls)
            candidates.append((mats, ths))
    
    # Multi-quarter-wave stacks with different center wavelengths
    center_wls_options = [
        [500, 600, 700],
        [450, 550, 650],
        [500, 550, 600],
        [400, 600, 800],
        [420, 520, 620, 720],
        [480, 580, 680],
    ]
    for n_layers in [6, 8, 10, 12, 14]:
        if n_layers > max_layers:
            continue
        for start_char in ['L', 'H']:
            for cw in center_wls_options:
                mats, ths = _multi_quarter_wave_stack(n_layers, start_char, n_L, n_H, n_ambient, n_substrate, wls, cw)
                candidates.append((mats, ths))
    
    # Random stacks
    for _ in range(12):
        n_layers = random.randint(6, min(20, max_layers))
        mats, ths = _random_stack(n_layers, max_layer, max_total)
        candidates.append((mats, ths))
    
    # Smooth profile-based initial guesses
    if time.time() < deadline - 30:
        try:
            mats_smooth, ths_smooth = _optimize_smooth_profile(problem, wls, n_mat, deadline - 20, max_layer, max_total)
            if mats_smooth:
                candidates.append((mats_smooth, ths_smooth))
        except Exception:
            pass
    
    # Shuffle candidates
    random.shuffle(candidates)
    
    # Optimize each candidate, prioritizing promising ones
    # First pass: quick evaluation to rank candidates
    ranked_candidates = []
    for mats, ths in candidates:
        if time.time() > deadline - 30:
            break
        ths_arr = np.array(ths, dtype=float)
        ths_arr = _project_to_feasible(ths_arr, max_layer, max_total)
        val = _mean_reflectance_percent(mats, list(ths_arr), problem, wls, n_mat)
        ranked_candidates.append((val, mats, ths))
    
    # Sort by initial quality (best first)
    ranked_candidates.sort(key=lambda x: x[0])
    
    # Optimize top candidates with more time
    time_per_candidate = max(5, (deadline - time.time() - 15) / max(1, len(ranked_candidates[:8])))
    
    for val, mats, ths in ranked_candidates[:8]:
        if time.time() > deadline - 15:
            break
        
        # Allocate more time to better candidates
        time_budget = min(time_per_candidate * (1 + 0.5 * (1 - val/10)), deadline - time.time() - 10)
        
        x_opt = _optimize_stack(mats, ths, problem, wls, n_mat, time.time() + time_budget, max_layer, max_total)
        
        # Update best
        update_best(mats, x_opt)
        
        # Also try with the original thicknesses in case optimization failed
        update_best(mats, ths)
    
    # If time remains, try a few more candidates with less time
    for val, mats, ths in ranked_candidates[8:12]:
        if time.time() > deadline - 10:
            break
        
        x_opt = _optimize_stack(mats, ths, problem, wls, n_mat, deadline - 5, max_layer, max_total)
        update_best(mats, x_opt)
    
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
