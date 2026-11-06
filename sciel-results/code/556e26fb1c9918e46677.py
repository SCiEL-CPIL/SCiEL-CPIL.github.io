"""Broadband antireflection coating design via Differential Evolution and local polish."""
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

def _nelder_mead(f, x0, bounds, max_iter=500, tol=1e-10, time_deadline=None):
    """Nelder-Mead simplex with bounds."""
    n = len(x0)
    lo_bounds = np.array([b[0] for b in bounds])
    hi_bounds = np.array([b[1] for b in bounds])
    x0 = np.clip(np.asarray(x0, dtype=float), lo_bounds, hi_bounds)
    
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    
    for i in range(n):
        simplex[i + 1] = x0.copy()
        lo, hi = bounds[i]
        step = 0.05 * (hi - lo) if hi > lo else 0.05
        simplex[i + 1][i] += step
    
    fvals = np.array([f(simplex[i]) for i in range(n + 1)])
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for it in range(max_iter):
        if time_deadline is not None and time.time() > time_deadline:
            break
        
        idx = np.argsort(fvals)
        simplex = simplex[idx]
        fvals = fvals[idx]
        
        if np.max(fvals) - np.min(fvals) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        x_r = centroid + alpha * (centroid - simplex[-1])
        x_r = np.clip(x_r, lo_bounds, hi_bounds)
        f_r = f(x_r)
        
        if f_r < fvals[0]:
            x_e = centroid + gamma * (x_r - centroid)
            x_e = np.clip(x_e, lo_bounds, hi_bounds)
            f_e = f(x_e)
            
            if f_e < f_r:
                simplex[-1] = x_e
                fvals[-1] = f_e
            else:
                simplex[-1] = x_r
                fvals[-1] = f_r
        elif f_r < fvals[-2]:
            simplex[-1] = x_r
            fvals[-1] = f_r
        else:
            x_c = centroid + rho * (simplex[-1] - centroid)
            x_c = np.clip(x_c, lo_bounds, hi_bounds)
            f_c = f(x_c)
            
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], lo_bounds, hi_bounds)
                    fvals[i] = f(simplex[i])
    
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]

def _differential_evolution(f, x0, bounds, max_iter=100, pop_size=50, time_deadline=None):
    """Differential Evolution for global optimization."""
    n = len(x0)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    
    # Initialize population
    pop = np.random.uniform(lo, hi, size=(pop_size, n))
    # Ensure x0 is in the population
    pop[0] = np.clip(x0, lo, hi)
    
    fvals = np.array([f(pop[i]) for i in range(pop_size)])
    
    best_idx = np.argmin(fvals)
    best_x = pop[best_idx].copy()
    best_f = fvals[best_idx]
    
    F = 0.8  # Scale factor
    CR = 0.9  # Crossover rate
    
    for it in range(max_iter):
        if time_deadline is not None and time.time() > time_deadline:
            break
        
        # Select best
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]
        
        for i in range(pop_size):
            # Select three distinct random indices
            r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            
            # Mutation
            v = pop[r1] + F * (pop[r2] - pop[r3])
            
            # Crossover
            jrand = np.random.randint(n)
            trial = np.empty(n)
            for j in range(n):
                if np.random.rand() < CR or j == jrand:
                    trial[j] = v[j]
                else:
                    trial[j] = pop[i][j]
            
            trial = np.clip(trial, lo, hi)
            
            # Selection
            f_trial = f(trial)
            if f_trial < fvals[i]:
                pop[i] = trial
                fvals[i] = f_trial
    
    best_idx = np.argmin(fvals)
    return pop[best_idx], fvals[best_idx]

def design(problem):
    """Design a broadband antireflection coating."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
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
    
    # Define high-priority patterns
    # We focus on odd numbers of layers which allow for symmetric AR designs
    # Prioritize 7, 9, 11, 13 layers
    patterns = []
    
    for n in [7, 9, 11, 13, 5, 15]:
        if n > max_layers:
            continue
        # Alternating L/H
        labels_lh = ['L' if i % 2 == 0 else 'H' for i in range(n)]
        patterns.append((n, labels_lh))
        
        # Alternating H/L
        labels_hl = ['H' if i % 2 == 0 else 'L' for i in range(n)]
        patterns.append((n, labels_hl))
        
    # Sort by priority (already sorted by construction, but explicit is better)
    # We will process them in the order defined above.
    
    # Time budget per pattern
    # Total 145s. 12 patterns. ~12s per pattern.
    # DE is fast. NM is slower.
    
    for n_layers, labels in patterns:
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
        
        # Initial guess: Quarter-wave stack
        x0 = np.array([center_wl / (4.0 * n) for n in indices])
        x0 = np.clip(x0, 0, max_layer_nm)
        if np.sum(x0) > max_total_nm:
            x0 = x0 * (max_total_nm / np.sum(x0))
        
        f0 = f(x0)
        if f0 < best_score:
            best_score = f0
            best_labels = labels.copy()
            best_thicknesses = x0.copy()
        
        if f0 >= 100.0:
            continue
        
        # Phase 1: Differential Evolution
        de_deadline = time.time() + 8.0
        try:
            x_de, f_de = _differential_evolution(f, x0, bounds, max_iter=50, pop_size=40, 
                                                 time_deadline=de_deadline)
        except Exception:
            x_de, f_de = x0, f0
        
        if f_de < best_score:
            best_score = f_de
            best_labels = labels.copy()
            best_thicknesses = x_de.copy()
        
        # Phase 2: Nelder-Mead Polish
        if time.time() < deadline - 5:
            nm_deadline = time.time() + 3.0
            try:
                x_nm, f_nm = _nelder_mead(f, x_de, bounds, max_iter=200, tol=1e-12, 
                                           time_deadline=nm_deadline)
            except Exception:
                x_nm, f_nm = x_de, f_de
            
            if f_nm < best_score:
                best_score = f_nm
                best_labels = labels.copy()
                best_thicknesses = x_nm.copy()
        
        # Early exit if very good
        if best_score < 0.1:
            break
    
    # Final fallback
    if best_labels is None:
        return ["L"], [center_wl / (4.0 * materials["L"])]
    
    # Ensure constraints on final result
    thicknesses = np.clip(best_thicknesses, 0, max_layer_nm)
    if np.sum(thicknesses) > max_total_nm:
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
    
    return best_labels, thicknesses.tolist()
