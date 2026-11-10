"""Broadband AR coating design with vectorized DE + Nelder-Mead refinement."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _reflectance_batch(materials, thicknesses, lambdas, n0, ns):
    """
    Compute reflectance for a batch of candidates, vectorized.
    materials: list of lists of indices, shape (batch, max_layers)
    thicknesses: list of arrays, shape (batch, max_layers)
    lambdas: array of wavelengths
    Returns: array of mean reflectance (fraction) per candidate
    """
    batch_size = len(thicknesses)
    max_layers = len(lambdas) if False else (len(thicknesses[0]) if batch_size > 0 else 0)
    M = len(lambdas)
    
    # Stack into arrays: (batch, max_layers)
    mats = np.array(materials, dtype=float)  # (batch, max_layers)
    thicks = np.array(thicknesses, dtype=float)  # (batch, max_layers)
    
    # Initialize total matrix: identity for each candidate
    M11 = np.ones((batch_size, M), dtype=complex)
    M12 = np.zeros((batch_size, M), dtype=complex)
    M21 = np.zeros((batch_size, M), dtype=complex)
    M22 = np.ones((batch_size, M), dtype=complex)
    
    for j in range(max_layers):
        nj = mats[:, j]  # (batch,)
        tj = thicks[:, j]  # (batch,)
        
        # d = 2*pi*n*t/lambda: shape (batch, M)
        d = 2.0 * np.pi * nj[:, np.newaxis] * tj[:, np.newaxis] / lambdas[np.newaxis, :]
        
        cd = np.cos(d)
        sd = np.sin(d)
        
        i_sd_nj = 1j * sd / nj[:, np.newaxis]
        i_nj_sd = 1j * nj[:, np.newaxis] * sd
        
        new_M11 = M11 * cd + M12 * i_sd_nj
        new_M12 = M11 * i_sd_nj + M12 * cd
        new_M21 = M21 * cd + M22 * i_nj_sd
        new_M22 = M21 * i_nj_sd + M22 * cd
        
        M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
    
    B = M11 + M12 * ns
    C = M21 + M22 * ns
    Y = C / B
    r = (n0 - Y) / (n0 + Y)
    R = np.abs(r)**2  # (batch, M)
    
    return np.mean(R, axis=1)  # (batch,)

def _reflectance_single(materials, thicknesses, lambdas, n0, ns):
    """Compute mean reflectance for a single candidate."""
    mats = [materials]
    thicks = [thicknesses]
    return _reflectance_batch(mats, thicks, lambdas, n0, ns)[0]

def _decode_candidate(params, max_layers, max_layer_nm, max_total_nm, mat_L, mat_H):
    """
    Decode a parameter vector into materials and thicknesses.
    params: array of length 2*max_layers
    First max_layers elements: material codes (0=L, 1=H)
    Next max_layers elements: thicknesses
    Returns: (materials_list, thicknesses_list, valid)
    """
    mat_codes = params[:max_layers]
    thicks = params[max_layers:]
    
    # Determine number of layers: find last non-zero thickness
    # But we need to handle material codes too
    # A layer is "active" if its thickness > 0.1 nm
    active = thicks > 0.1
    if not np.any(active):
        return None, None, False
    
    last_active = np.where(active)[0][-1]
    n_layers = last_active + 1
    
    materials = []
    thicknesses = []
    for i in range(n_layers):
        if mat_codes[i] < 0.5:
            materials.append(mat_L)
        else:
            materials.append(mat_H)
        thicknesses.append(thicks[i])
    
    # Pad to max_layers for batch evaluation
    while len(materials) < max_layers:
        materials.append(mat_L)
        thicknesses.append(0.0)
    
    total = sum(thicknesses[:n_layers])
    if total > max_total_nm:
        return None, None, False
    
    for t in thicknesses[:n_layers]:
        if t > max_layer_nm:
            return None, None, False
    
    return materials, thicknesses, True

def _objective_batch(params_batch, max_layers, max_layer_nm, max_total_nm, mat_L, mat_H, lambdas, n0, ns):
    """
    Evaluate a batch of candidates.
    params_batch: (batch, 2*max_layers)
    Returns: array of objective values (mean reflectance in %)
    """
    batch_size = params_batch.shape[0]
    results = np.full(batch_size, 100.0)  # default high value
    
    valid_materials = []
    valid_thicks = []
    valid_indices = []
    
    for i in range(batch_size):
        mats, thicks, valid = _decode_candidate(
            params_batch[i], max_layers, max_layer_nm, max_total_nm, mat_L, mat_H
        )
        if valid:
            valid_materials.append(mats)
            valid_thicks.append(thicks)
            valid_indices.append(i)
    
    if valid_materials:
        R_vals = _reflectance_batch(valid_materials, valid_thicks, lambdas, n0, ns)
        for idx, val in zip(valid_indices, R_vals):
            results[idx] = val * 100.0
    
    return results

def _nelder_mead(objective, x0, bounds, max_iter=200, time_limit=None, start_time=None):
    """
    Nelder-Mead simplex optimization.
    objective: function taking 1D array, returns scalar
    x0: initial point
    bounds: list of (low, high)
    """
    x0 = np.array(x0, dtype=float)
    n = len(x0)
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
    # Initialize simplex
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        step = 0.1 * (upper[i] - lower[i]) if upper[i] != lower[i] else 0.1
        xi[i] += step
        xi = np.clip(xi, lower, upper)
        simplex.append(xi)
    
    fvals = np.array([objective(x) for x in simplex])
    best_val = np.min(fvals)
    best_x = simplex[np.argmin(fvals)].copy()
    
    alpha = 1.0   # reflection
    gamma = 2.0   # expansion
    rho = 0.5     # contraction
    sigma = 0.5   # shrink
    
    for it in range(max_iter):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Sort simplex by objective value
        order = np.argsort(fvals)
        simplex = [simplex[i].copy() for i in order]
        fvals = fvals[order].copy()
        
        # Check convergence
        if np.std(fvals) < 1e-8:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        xr = np.clip(xr, lower, upper)
        fr = objective(xr)
        
        if fr < fvals[0]:
            # Try expansion
            xe = centroid + gamma * (xr - centroid)
            xe = np.clip(xe, lower, upper)
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            if fr < fvals[-1]:
                # Outside contraction
                xc = centroid + rho * (xr - centroid)
            else:
                # Inside contraction
                xc = centroid - rho * (centroid - simplex[-1])
            xc = np.clip(xc, lower, upper)
            fc = objective(xc)
            
            if fc < min(fr, fvals[-1]):
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], lower, upper)
                    fvals[i] = objective(simplex[i])
        
        # Update best
        min_idx = np.argmin(fvals)
        if fvals[min_idx] < best_val:
            best_val = fvals[min_idx]
            best_x = simplex[min_idx].copy()
    
    return best_x, best_val

def design(problem):
    """Design broadband AR coating."""
    t_start = time.time()
    deadline = t_start + 135.0  # safety margin
    
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    mat_L = problem["materials"]["L"]
    mat_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    lambdas = np.linspace(band[0], band[1], 401)
    
    # Parameter encoding:
    # First max_layers: material codes (0=L, 1=H)
    # Next max_layers: thicknesses
    # Total: 2 * max_layers
    D = 2 * max_layers
    bounds = [(0.0, 1.0)] * max_layers + [(0.0, max_layer_nm)] * max_layers
    
    def _obj_single(params):
        mats, thicks, valid = _decode_candidate(
            params, max_layers, max_layer_nm, max_total_nm, mat_L, mat_H
        )
        if not valid:
            return 100.0
        R = _reflectance_single(mats, thicks, lambdas, n0, ns)
        return R * 100.0
    
    # Vectorized Differential Evolution
    def _de_vectorized(popsize=40, maxiter=150, seed=42):
        np.random.seed(seed)
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        
        # Initialize population
        pop = lower + (upper - lower) * np.random.rand(popsize, D)
        # Ensure material codes are 0 or 1 (binary)
        pop[:, :max_layers] = np.round(pop[:, :max_layers])
        
        # Evaluate initial population
        fitness = _objective_batch(pop, max_layers, max_layer_nm, max_total_nm, mat_L, mat_H, lambdas, n0, ns)
        best_idx = np.argmin(fitness)
        best_individual = pop[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        F = 0.8
        CR = 0.9
        
        for it in range(maxiter):
            if time.time() > deadline - 30.0:  # reserve time for NM
                break
            
            # Generate all trials in batch
            trials = np.zeros_like(pop)
            for i in range(popsize):
                # Select 3 distinct random individuals
                candidates = [j for j in range(popsize) if j != i]
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
                
                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                trial = pop[i].copy()
                j_rand = np.random.randint(D)
                mask = np.random.rand(D) < CR
                mask[j_rand] = True
                trial[mask] = v[mask]
                
                # Bounds
                trial = np.clip(trial, lower, upper)
                # Binary material codes
                trial[:max_layers] = np.round(trial[:max_layers])
                
                trials[i] = trial
            
            # Evaluate all trials in batch
            trial_fitness = _objective_batch(trials, max_layers, max_layer_nm, max_total_nm, mat_L, mat_H, lambdas, n0, ns)
            
            # Selection
            improved = trial_fitness < fitness
            pop[improved] = trials[improved]
            fitness[improved] = trial_fitness[improved]
            
            # Update best
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fitness:
                best_fitness = fitness[min_idx]
                best_individual = pop[min_idx].copy()
        
        return best_individual, best_fitness
    
    # Run DE
    best_params, best_score = _de_vectorized(popsize=40, maxiter=150, seed=42)
    
    # Nelder-Mead refinement on the best DE result
    if time.time() < deadline - 15.0:
        # Create a local objective that handles the binary material codes
        # We'll relax the binary constraint slightly for NM to work better
        # Actually, let's keep binary but use a custom NM that respects it
        
        def _obj_nm(params):
            # Round material codes
            params_rounded = params.copy()
            params_rounded[:max_layers] = np.round(params_rounded[:max_layers])
            return _obj_single(params_rounded)
        
        # Use NM on a subset of parameters (thicknesses only, keep materials fixed)
        # Or do full NM with binary handling
        nm_start = time.time()
        nm_time_limit = min(25.0, deadline - nm_start - 5.0)
        
        if nm_time_limit > 5.0:
            # NM on thicknesses only, keeping best materials
            best_mats = np.round(best_params[:max_layers])
            best_thicks = best_params[max_layers:]
            
            def _obj_thicks(thick_params):
                full = np.concatenate([best_mats, thick_params])
                full = np.clip(full, 0.0, 1.0)
                full[:max_layers] = np.round(full[:max_layers])
                return _obj_single(full)
            
            thick_bounds = [(0.0, max_layer_nm)] * max_layers
            x0_thick = best_thicks.copy()
            
            try:
                best_thick, best_thick_val = _nelder_mead(
                    _obj_thicks, x0_thick, thick_bounds,
                    max_iter=100, time_limit=nm_time_limit, start_time=nm_start
                )
                if best_thick_val < best_score:
                    best_params[max_layers:] = best_thick
                    best_score = best_thick_val
            except Exception:
                pass
            
            # Also try NM with small material perturbations
            if time.time() < deadline - 10.0:
                # Try flipping each material position and see if it helps
                for flip_i in range(max_layers):
                    if time.time() > deadline - 10.0:
                        break
                    test_params = best_params.copy()
                    test_params[flip_i] = 1.0 - test_params[flip_i]
                    test_val = _obj_single(test_params)
                    if test_val < best_score:
                        best_params = test_params
                        best_score = test_val
    
    # Extract final solution
    n_layers = 0
    materials_list = []
    thicknesses_list = []
    
    for i in range(max_layers):
        t = best_params[max_layers + i]
        if t > 0.5:  # threshold for active layer
            mat_code = best_params[i]
            if mat_code < 0.5:
                materials_list.append("L")
            else:
                materials_list.append("H")
            thicknesses_list.append(float(t))
            n_layers = i + 1
    
    if n_layers == 0:
        # Fallback: quarter-wave stack
        centre = 0.5 * (band[0] + band[1])
        t_L = centre / (4.0 * mat_L)
        t_H = centre / (4.0 * mat_H)
        if t_L + t_H <= max_total_nm and t_L <= max_layer_nm and t_H <= max_layer_nm:
            return ["L", "H"], [t_L, t_H]
        else:
            return ["L"], [min(t_L, max_layer_nm)]
    
    # Verify and adjust
    total = sum(thicknesses_list)
    if total > max_total_nm:
        scale = max_total_nm / total
        thicknesses_list = [t * scale for t in thicknesses_list]
    
    for t in thicknesses_list:
        if t > max_layer_nm:
            thicknesses_list[thicknesses_list.index(t)] = max_layer_nm
    
    return materials_list, thicknesses_list

# EVOLVE-BLOCK-END
