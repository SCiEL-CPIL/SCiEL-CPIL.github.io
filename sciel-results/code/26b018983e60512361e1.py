"""Broadband antireflection coating design: smooth multi-start local optimization."""
import numpy as np
import time

def _compute_reflectance(wavelengths_nm, indices, thicknesses_nm, ambient_idx, substrate_idx):
    """Compute reflectance (fraction, 0-1) at each wavelength using characteristic matrix method."""
    n_layers = len(indices)
    if n_layers == 0:
        r0 = (ambient_idx - substrate_idx) / (ambient_idx + substrate_idx)
        return np.full(len(wavelengths_nm), r0 * r0)
    
    k = 2.0 * np.pi / wavelengths_nm
    
    M00 = np.ones(len(wavelengths_nm), dtype=complex)
    M01 = np.zeros(len(wavelengths_nm), dtype=complex)
    M10 = np.zeros(len(wavelengths_nm), dtype=complex)
    M11 = np.ones(len(wavelengths_nm), dtype=complex)
    
    for i in range(n_layers):
        n_i = indices[i]
        d_i = thicknesses_nm[i]
        delta = k * n_i * d_i
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        B00 = cos_d
        B01 = 1j * sin_d / n_i
        B10 = 1j * n_i * sin_d
        B11 = cos_d
        
        new_M00 = M00 * B00 + M01 * B10
        new_M01 = M00 * B01 + M01 * B11
        new_M10 = M10 * B00 + M11 * B10
        new_M11 = M10 * B01 + M11 * B11
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    Y_out = substrate_idx
    Y_in = (M10 + M11 * Y_out) / (M00 + M01 * Y_out)
    
    r = (ambient_idx - Y_in) / (ambient_idx + Y_in)
    R = np.abs(r) ** 2
    
    return R

def _objective(thicknesses, indices, wavelengths_nm, ambient_idx, substrate_idx, 
               max_layer_nm, max_total_nm):
    """Objective: mean reflectance in percent."""
    thicknesses = np.asarray(thicknesses, dtype=float)
    if np.any(thicknesses < -1e-6) or np.any(thicknesses > max_layer_nm + 1e-6):
        return 100.0
    if np.sum(thicknesses) > max_total_nm + 1e-6:
        return 100.0
    
    thicknesses_clipped = np.clip(thicknesses, 0.0, max_layer_nm)
    if np.sum(thicknesses_clipped) > max_total_nm:
        return 100.0
    
    R = _compute_reflectance(wavelengths_nm, indices, thicknesses_clipped, 
                             ambient_idx, substrate_idx)
    return np.mean(R) * 100.0

def _nelder_mead(f, x0, bounds, max_iter=200, time_deadline=None):
    """
    Nelder-Mead simplex optimization with bounds.
    f: objective function
    x0: initial guess
    bounds: list of (min, max) tuples
    max_iter: maximum iterations
    time_deadline: absolute time limit
    """
    n = len(x0)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    
    # Initialize simplex
    simplex = [np.clip(x0, lo, hi)]
    for i in range(n):
        x_new = simplex[0].copy()
        if x_new[i] + 0.1 > hi[i]:
            x_new[i] = hi[i] - 0.01
        else:
            x_new[i] += 0.1
        x_new = np.clip(x_new, lo, hi)
        simplex.append(x_new)
    
    simplex_f = [f(x) for x in simplex]
    
    alpha = 1.0  # reflection
    gamma = 2.0  # expansion
    rho = 0.5    # contraction
    sigma = 0.5  # shrink
    
    for it in range(max_iter):
        if time_deadline is not None and time.time() > time_deadline:
            break
            
        # Sort simplex by function value
        idx = np.argsort(simplex_f)
        simplex = [simplex[i] for i in idx]
        simplex_f = [simplex_f[i] for i in idx]
        
        # Check convergence
        if abs(simplex_f[0] - simplex_f[-1]) < 1e-8 * (1 + abs(simplex_f[0])):
            break
            
        # Centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = centroid + alpha * (centroid - simplex[-1])
        x_r = np.clip(x_r, lo, hi)
        f_r = f(x_r)
        
        if f_r < simplex_f[0]:
            # Expansion
            x_e = centroid + gamma * (x_r - centroid)
            x_e = np.clip(x_e, lo, hi)
            f_e = f(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                simplex_f[-1] = f_e
            else:
                simplex[-1] = x_r
                simplex_f[-1] = f_r
        elif f_r < simplex_f[-2]:
            # Reflection is good but not best
            simplex[-1] = x_r
            simplex_f[-1] = f_r
        else:
            # Contraction
            x_c = centroid + rho * (simplex[-1] - centroid)
            x_c = np.clip(x_c, lo, hi)
            f_c = f(x_c)
            
            if f_c < simplex_f[-1]:
                simplex[-1] = x_c
                simplex_f[-1] = f_c
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], lo, hi)
                    simplex_f[i] = f(simplex[i])
    
    # Return best point
    best_idx = np.argmin(simplex_f)
    return simplex[best_idx], simplex_f[best_idx]

def design(problem):
    """Design a broadband antireflection coating."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wavelengths_nm = np.linspace(band_nm[0], band_nm[1], 401)
    center_wl = 0.5 * (band_nm[0] + band_nm[1])
    
    best_labels = None
    best_thicknesses = None
    best_score = np.inf
    
    # Generate layer patterns - focus on odd numbers of layers for symmetric designs
    patterns = []
    
    # Alternating starting with L (most common for AR coatings)
    for n in range(3, min(max_layers + 1, 21), 2):
        labels = ['L' if i % 2 == 0 else 'H' for i in range(n)]
        patterns.append((n, labels, "alt_L"))
    
    # Alternating starting with H
    for n in range(3, min(max_layers + 1, 21), 2):
        labels = ['H' if i % 2 == 0 else 'L' for i in range(n)]
        patterns.append((n, labels, "alt_H"))
    
    # Prioritize: 5, 7, 9 layers first (best tradeoff between complexity and performance)
    def pattern_priority(p):
        n = p[0]
        if n in [5, 7, 9]:
            return 0
        elif n in [3, 11]:
            return 1
        elif n in [13, 15]:
            return 2
        else:
            return 3
    
    patterns.sort(key=pattern_priority)
    
    # For each pattern, do multi-start optimization
    for n_layers, labels, ptype in patterns:
        if time.time() > deadline - 10:
            break
        
        indices = [materials[label] for label in labels]
        bounds = [(0.0, float(max_layer_nm))] * n_layers
        
        def make_f(indices):
            def f(x):
                return _objective(x, indices, wavelengths_nm, ambient_idx, substrate_idx,
                                  max_layer_nm, max_total_nm)
            return f
        
        f = make_f(indices)
        
        # Multiple initial guesses
        n_starts = 5
        for start_i in range(n_starts):
            if time.time() > deadline - 5:
                break
            
            # Generate initial guess based on physics
            if start_i == 0:
                # Quarter-wave stack at center wavelength
                x0 = np.array([center_wl / (4.0 * n) for n in indices])
            elif start_i == 1:
                # Quarter-wave at shorter wavelength (toward blue)
                x0 = np.array([band_nm[0] / (4.0 * n) for n in indices])
            elif start_i == 2:
                # Quarter-wave at longer wavelength (toward red)
                x0 = np.array([band_nm[1] / (4.0 * n) for n in indices])
            elif start_i == 3:
                # Randomized quarter-wave
                x0 = np.array([center_wl / (4.0 * n) for n in indices])
                x0 = x0 * np.random.uniform(0.7, 1.3, size=n_layers)
            else:
                # Another random start
                x0 = np.random.uniform(0, max_layer_nm, size=n_layers)
                x0 = x0 * (max_total_nm / (n_layers * max_layer_nm)) # scale to total budget
            
            x0 = np.clip(x0, 0, max_layer_nm)
            if np.sum(x0) > max_total_nm:
                x0 = x0 * (max_total_nm / np.sum(x0))
            
            # Evaluate initial
            f0 = f(x0)
            if f0 < best_score:
                best_score = f0
                best_labels = labels.copy()
                best_thicknesses = x0.copy()
            
            if f0 >= 100.0:
                continue
            
            # Run Nelder-Mead
            opt_deadline = time.time() + min(10.0, deadline - 5 - time.time())
            try:
                x_opt, f_opt = _nelder_mead(f, x0, bounds, max_iter=100, time_deadline=opt_deadline)
            except Exception:
                x_opt, f_opt = x0, f0
            
            if f_opt < best_score:
                best_score = f_opt
                best_labels = labels.copy()
                best_thicknesses = x_opt.copy()
            
            # Early exit if very good
            if best_score < 0.5:
                break
        
        # Early exit if very good
        if best_score < 0.5:
            break
    
    # Final fallback
    if best_labels is None:
        return ["L"], [center_wl / (4.0 * materials["L"])]
    
    # Ensure constraints on final result
    thicknesses = np.clip(best_thicknesses, 0, max_layer_nm)
    if np.sum(thicknesses) > max_total_nm:
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
    
    return best_labels, thicknesses.tolist()
