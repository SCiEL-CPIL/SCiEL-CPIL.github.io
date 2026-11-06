"""Broadband antireflection coating design via differential evolution + Nelder-Mead polish."""
import numpy as np
import time
import math

def _compute_reflectance(wavelengths_nm, indices, thicknesses_nm, ambient_idx, substrate_idx):
    """
    Compute reflectance (fraction, 0-1) at each wavelength using the characteristic matrix method.
    Vectorized over wavelengths.
    """
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

def _objective(params, indices, wavelengths_nm, ambient_idx, substrate_idx, max_layer_nm, max_total_nm):
    """
    Objective function: mean reflectance in percent with penalty for constraint violations.
    """
    thicknesses = np.clip(params, 0, max_layer_nm)
    
    total = np.sum(thicknesses)
    if total > max_total_nm:
        penalty = (total - max_total_nm) * 0.1
        return 100.0 + penalty
    
    R = _compute_reflectance(wavelengths_nm, indices, thicknesses, ambient_idx, substrate_idx)
    return np.mean(R) * 100.0

def _nelder_mead(f, x0, bounds, maxiter=2000, xatol=1e-6, fatol=1e-8):
    """
    Simplex-based Nelder-Mead optimizer with bounds enforcement.
    """
    n = len(x0)
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    # Initialize simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        xi = x0.copy()
        xi[i] += 0.05 * (bounds[i][1] - bounds[i][0]) if bounds[i][1] > bounds[i][0] else 0.05
        simplex.append(xi)
    
    # Enforce bounds
    for i in range(len(simplex)):
        for j in range(n):
            simplex[i][j] = np.clip(simplex[i][j], bounds[j][0], bounds[j][1])
    
    # Evaluate
    for i in range(len(simplex)):
        simplex[i] = np.concatenate([simplex[i], [f(simplex[i][:n])]])
    
    for _ in range(maxiter):
        # Sort by function value
        simplex.sort(key=lambda x: x[-1])
        
        # Check convergence
        if abs(simplex[0][-1] - simplex[-1][-1]) < fatol:
            break
        if np.max(np.abs(simplex[0][:n] - simplex[-1][:n])) < xatol:
            break
        
        centroid = np.mean([simplex[i][:n] for i in range(len(simplex) - 1)], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1][:n])
        xr = np.clip(xr, [b[0] for b in bounds], [b[1] for b in bounds])
        fr = f(xr)
        
        if fr < simplex[0][-1]:
            # Expansion
            xe = centroid + gamma * (xr - centroid)
            xe = np.clip(xe, [b[0] for b in bounds], [b[1] for b in bounds])
            fe = f(xe)
            if fe < fr:
                simplex[-1] = np.concatenate([xe, [fe]])
            else:
                simplex[-1] = np.concatenate([xr, [fr]])
        elif fr < simplex[-2][-1]:
            simplex[-1] = np.concatenate([xr, [fr]])
        else:
            # Contraction
            xc = centroid + rho * (simplex[-1][:n] - centroid)
            xc = np.clip(xc, [b[0] for b in bounds], [b[1] for b in bounds])
            fc = f(xc)
            if fc < simplex[-1][-1]:
                simplex[-1] = np.concatenate([xc, [fc]])
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i][:n] = simplex[0][:n] + sigma * (simplex[i][:n] - simplex[0][:n])
                    simplex[i][:n] = np.clip(simplex[i][:n], [b[0] for b in bounds], [b[1] for b in bounds])
                    simplex[i][-1] = f(simplex[i][:n])
    
    simplex.sort(key=lambda x: x[-1])
    return simplex[0][:n], simplex[0][-1]

def _differential_evolution(f, bounds, pop_size=50, maxiter=100, F=0.8, CR=0.9, seed=42, time_limit=None, start_time=None):
    """
    Differential evolution optimizer.
    """
    np.random.seed(seed)
    n = len(bounds)
    
    # Initialize population
    pop = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], size=(pop_size, n))
    fitness = np.array([f(ind) for ind in pop])
    
    best_idx = np.argmin(fitness)
    best_x = pop[best_idx].copy()
    best_f = fitness[best_idx]
    
    for gen in range(maxiter):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        for i in range(pop_size):
            # Select three distinct random individuals
            a, b, c = np.random.choice(pop_size, size=3, replace=False)
            while a == b or a == c or b == c:
                a, b, c = np.random.choice(pop_size, size=3, replace=False)
            
            # Create trial vector
            v = pop[a] + F * (pop[b] - pop[c])
            v = np.clip(v, [bd[0] for bd in bounds], [bd[1] for bd in bounds])
            
            # Crossover
            trial = pop[i].copy()
            j_rand = np.random.randint(n)
            for j in range(n):
                if np.random.rand() < CR or j == j_rand:
                    trial[j] = v[j]
            
            trial = np.clip(trial, [bd[0] for bd in bounds], [bd[1] for bd in bounds])
            
            # Selection
            ft = f(trial)
            if ft <= fitness[i]:
                pop[i] = trial
                fitness[i] = ft
                if ft < best_f:
                    best_f = ft
                    best_x = trial.copy()
    
    return best_x, best_f

def _polish(x0, f, bounds, time_limit=None, start_time=None):
    """
    Polish a solution using Nelder-Mead.
    """
    if time_limit is not None and start_time is not None:
        if time.time() - start_time > time_limit:
            return x0, f(x0)
    
    x, fx = _nelder_mead(f, x0, bounds, maxiter=1000, xatol=1e-7, fatol=1e-9)
    return x, fx

def design(problem):
    """
    Design a broadband antireflection coating.
    Returns: (labels, thicknesses)
    """
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
    
    best_result = None
    best_score = np.inf
    
    # Define candidate label patterns to try
    candidates = []
    
    # Single layer
    candidates.append(['L'])
    candidates.append(['H'])
    
    # 3 layers
    candidates.append(['L', 'H', 'L'])
    candidates.append(['H', 'L', 'H'])
    
    # 5 layers
    candidates.append(['L', 'H', 'L', 'H', 'L'])
    candidates.append(['H', 'L', 'H', 'L', 'H'])
    
    # 7 layers
    candidates.append(['L', 'H', 'L', 'H', 'L', 'H', 'L'])
    candidates.append(['H', 'L', 'H', 'L', 'H', 'L', 'H'])
    
    # 9 layers
    candidates.append(['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'])
    candidates.append(['H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'])
    
    # 11 layers
    candidates.append(['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'])
    candidates.append(['H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'])
    
    # 13 layers
    candidates.append(['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'])
    candidates.append(['H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'])
    
    # 15 layers
    candidates.append(['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'])
    candidates.append(['H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'])
    
    # 17 layers
    candidates.append(['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'])
    candidates.append(['H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'])
    
    # 19 layers
    candidates.append(['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'])
    candidates.append(['H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'])
    
    # Filter candidates that fit within max_layers
    candidates = [c for c in candidates if len(c) <= max_layers]
    
    # Try each candidate with DE + polish
    for labels in candidates:
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            break
        
        # Allocate time budget based on remaining time
        remaining_candidates = len(candidates) - candidates.index(labels)
        remaining_time = deadline - 5 - elapsed
        time_budget = max(3.0, remaining_time / remaining_candidates)
        
        try:
            indices = [materials[label] for label in labels]
            n_layers = len(labels)
            
            center_wl = 0.5 * (band_nm[0] + band_nm[1])
            
            # Initial guess: quarter-wave
            x0 = np.array([center_wl / (4.0 * n) for n in indices])
            
            # Bounds
            bounds = [(0, max_layer_nm)] * n_layers
            
            def obj(x):
                return _objective(x, indices, wavelengths_nm, ambient_idx, substrate_idx, max_layer_nm, max_total_nm)
            
            # Evaluate initial
            f0 = obj(x0)
            
            # Differential evolution
            pop_size = min(30, max(10, n_layers * 2))
            de_time_limit = time_budget * 0.6
            de_start = time.time()
            
            best_x, best_f = _differential_evolution(
                obj, bounds, pop_size=pop_size, maxiter=50,
                F=0.8, CR=0.9, seed=42,
                time_limit=de_time_limit, start_time=de_start
            )
            
            # Polish with Nelder-Mead
            polish_time_limit = time_budget * 0.4
            polish_start = time.time()
            
            best_x, best_f = _polish(best_x, obj, bounds,
                                     time_limit=polish_time_limit, start_time=polish_start)
            
            # Clip and check constraints
            best_x = np.clip(best_x, 0, max_layer_nm)
            if np.sum(best_x) > max_total_nm:
                best_x = best_x * (max_total_nm / np.sum(best_x))
            
            # Final evaluation
            final_f = obj(best_x)
            
            if final_f < best_score:
                best_score = final_f
                best_result = (labels, best_x.tolist())
            
            # Early termination if we're doing very well
            if best_score < 0.3:
                break
                
        except Exception:
            continue
    
    # If no result found, fall back to single quarter-wave layer
    if best_result is None:
        center_wl = 0.5 * (band_nm[0] + band_nm[1])
        return ["L"], [center_wl / (4.0 * materials["L"])]
    
    labels, thicknesses = best_result
    
    # Final constraint enforcement
    thicknesses = [max(0, min(t, max_layer_nm)) for t in thicknesses]
    total = sum(thicknesses)
    if total > max_total_nm:
        scale = max_total_nm / total
        thicknesses = [t * scale for t in thicknesses]
    
    return labels, thicknesses
