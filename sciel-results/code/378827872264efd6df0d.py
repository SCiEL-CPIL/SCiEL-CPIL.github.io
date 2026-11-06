"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

# EVOLVE-BLOCK-START

def _reflectance_vectorized(materials, thicknesses, wls, ambient_index=1.0, substrate_index=1.52, materials_dict=None):
    """
    Compute reflectance at all wavelengths using the characteristic matrix method (vectorized).
    """
    if materials_dict is None:
        materials_dict = {'L': 1.38, 'H': 2.35}
    
    n_layers = len(materials)
    if n_layers == 0:
        r = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return np.full_like(wls, r*r)
        
    M11 = np.ones_like(wls, dtype=complex)
    M12 = np.zeros_like(wls, dtype=complex)
    M21 = np.zeros_like(wls, dtype=complex)
    M22 = np.ones_like(wls, dtype=complex)
    
    for mat, thick in zip(materials, thicknesses):
        n = materials_dict[mat]
        if thick < 1e-9:
            continue
            
        beta = 2.0 * np.pi * n * thick / wls
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        L11 = cos_b
        L12 = 1.0j * sin_b / n
        L21 = n * 1.0j * sin_b
        L22 = cos_b
        
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        
        M11 = new_M11
        M12 = new_M12
        M21 = new_M21
        M22 = new_M22
        
    Y_s = substrate_index
    Y = (M21 + M22 * Y_s) / (M11 + M12 * Y_s)
    
    Y_0 = ambient_index
    r = (Y_0 - Y) / (Y_0 + Y)
    R = np.abs(r) ** 2
    return R

def _mean_reflectance(materials, thicknesses, band_nm=(400, 800), n_points=401,
                       ambient_index=1.0, substrate_index=1.52, materials_dict=None):
    """Compute mean reflectance in percent over the band."""
    if materials_dict is None:
        materials_dict = {'L': 1.38, 'H': 2.35}
    
    wls = np.linspace(band_nm[0], band_nm[1], n_points)
    refls = _reflectance_vectorized(materials, thicknesses, wls, ambient_index, substrate_index, materials_dict)
    return 100.0 * np.mean(refls)

def _is_valid(materials, thicknesses, problem):
    """Check if the design satisfies all constraints."""
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    if len(materials) > max_layers:
        return False
    if len(materials) != len(thicknesses):
        return False
    for t in thicknesses:
        if t < 0 or t > max_layer_nm:
            return False
    if sum(thicknesses) > max_total_nm:
        return False
    return True

def _penalized_score(materials, thicknesses, problem, wls, materials_dict, 
                     ambient_index, substrate_index, band_nm):
    """Compute penalized mean reflectance. Returns a large value if constraints are violated."""
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    penalty = 0.0
    if len(materials) > max_layers:
        penalty += 10.0 * (len(materials) - max_layers)
    for t in thicknesses:
        if t < 0:
            penalty += 10.0 * abs(t)
        elif t > max_layer_nm:
            penalty += 10.0 * (t - max_layer_nm)
    total = sum(thicknesses)
    if total > max_total_nm:
        penalty += 10.0 * (total - max_total_nm)
    
    if penalty > 0:
        return 100.0 + penalty
    
    return _mean_reflectance(materials, thicknesses, band_nm, 401, ambient_index, substrate_index, materials_dict)

def _sa_step(materials, thicknesses, problem, wls, materials_dict, 
             ambient_index, substrate_index, band_nm, T, rng):
    """
    Perform one simulated annealing step. Returns new materials, thicknesses, and whether it was accepted.
    Mutates in place for speed, but returns copies for the caller.
    """
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    n_layers = len(materials)
    
    # Choose a move type
    # 1: Perturb thickness, 2: Flip layer type, 3: Add layer, 4: Remove layer, 5: Swap adjacent layers
    if n_layers < 2:
        move_type = rng.choice([1, 1, 1, 2, 3])  # More thickness perturbations early
    elif n_layers >= max_layers:
        move_type = rng.choice([1, 1, 2, 4, 5])  # Can't add
    else:
        move_type = rng.choice([1, 1, 1, 2, 3, 4, 5])
    
    # Current score
    current_score = _penalized_score(materials, thicknesses, problem, wls, materials_dict,
                                     ambient_index, substrate_index, band_nm)
    
    # Apply move
    new_materials = list(materials)
    new_thicknesses = list(thicknesses)
    
    if move_type == 1:  # Perturb a random thickness
        if n_layers == 0:
            return materials, thicknesses, False
        idx = rng.randint(0, n_layers - 1)
        # Gaussian perturbation scaled by current thickness or a base value
        base = max(thicknesses[idx], 50.0)
        delta = rng.gauss(0, base * 0.3)
        new_thicknesses[idx] = max(0.0, min(max_layer_nm, thicknesses[idx] + delta))
        
    elif move_type == 2:  # Flip a random layer type
        if n_layers == 0:
            return materials, thicknesses, False
        idx = rng.randint(0, n_layers - 1)
        new_materials[idx] = 'H' if materials[idx] == 'L' else 'L'
        
    elif move_type == 3:  # Add a layer at a random position
        if n_layers >= max_layers:
            return materials, thicknesses, False
        pos = rng.randint(0, n_layers)
        mat = rng.choice(['L', 'H'])
        n_mat = materials_dict[mat]
        # Initialize with a quarter-wave thickness at center wavelength
        wl_center = 0.5 * (band_nm[0] + band_nm[1])
        thick = wl_center / (4.0 * n_mat)
        # Add some randomness
        thick = max(10.0, min(max_layer_nm, thick * rng.uniform(0.5, 1.5)))
        new_materials.insert(pos, mat)
        new_thicknesses.insert(pos, thick)
        
    elif move_type == 4:  # Remove a random layer
        if n_layers <= 1:
            return materials, thicknesses, False
        idx = rng.randint(0, n_layers - 1)
        new_materials.pop(idx)
        new_thicknesses.pop(idx)
        
    elif move_type == 5:  # Swap adjacent layers
        if n_layers < 2:
            return materials, thicknesses, False
        idx = rng.randint(0, n_layers - 2)
        new_materials[idx], new_materials[idx+1] = new_materials[idx+1], new_materials[idx]
        new_thicknesses[idx], new_thicknesses[idx+1] = new_thicknesses[idx+1], new_thicknesses[idx]
    
    # New score
    new_score = _penalized_score(new_materials, new_thicknesses, problem, wls, materials_dict,
                                 ambient_index, substrate_index, band_nm)
    
    # Acceptance criterion
    if new_score < current_score:
        return new_materials, new_thicknesses, True
    else:
        delta_e = new_score - current_score
        if delta_e == 0:
            return new_materials, new_thicknesses, True
        # Metropolis criterion
        if T > 1e-8:
            prob = np.exp(-delta_e / T)
            if rng.random() < prob:
                return new_materials, new_thicknesses, True
    
    return materials, thicknesses, False

def _local_refine(materials, thicknesses, problem, wls, materials_dict, 
                  ambient_index, substrate_index, band_nm, max_iter=100):
    """
    Simple coordinate descent refinement for a fixed material sequence.
    """
    n_layers = len(materials)
    if n_layers == 0:
        return materials, thicknesses
    
    max_layer_nm = problem.get("max_layer_nm", 400)
    max_total_nm = problem.get("max_total_nm", 1000)
    
    current_score = _mean_reflectance(materials, thicknesses, band_nm, 401, 
                                       ambient_index, substrate_index, materials_dict)
    
    for _ in range(max_iter):
        improved = False
        for i in range(n_layers):
            # Try to optimize thickness[i] by ternary search or simple perturbation
            # Use a small grid search around current value
            t0 = thicknesses[i]
            best_t = t0
            best_s = current_score
            
            # Check bounds considering total thickness constraint
            other_sum = sum(thicknesses) - t0
            min_t = max(0.0, 0.0)
            max_t = min(max_layer_nm, max_total_nm - other_sum)
            
            if max_t <= min_t:
                continue
                
            # Grid search
            n_grid = 20
            grid = np.linspace(min_t, max_t, n_grid)
            for t_new in grid:
                # Quick check: if total thickness would exceed limit, skip
                if sum(thicknesses) - t0 + t_new > max_total_nm + 1e-6:
                    continue
                thicknesses[i] = t_new
                s = _mean_reflectance(materials, thicknesses, band_nm, 401, 
                                       ambient_index, substrate_index, materials_dict)
                if s < best_s:
                    best_s = s
                    best_t = t_new
            
            thicknesses[i] = best_t
            if best_s < current_score:
                current_score = best_s
                improved = True
        
        if not improved:
            break
    
    return materials, thicknesses

def design(problem):
    """
    Design using simulated annealing over both layer sequence and thicknesses.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    materials_dict = problem["materials"]
    nL = materials_dict["L"]
    nH = materials_dict["H"]
    n_sub = problem["substrate_index"]
    n_amb = problem["ambient_index"]
    band_nm = problem["band_nm"]
    wl_center = 0.5 * (band_nm[0] + band_nm[1])
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    wls = np.linspace(band_nm[0], band_nm[1], 401)
    
    rng = random.Random(42)
    
    # Initial solution: a few quarter-wave layers
    init_materials = []
    init_thicknesses = []
    # Start with a 3-layer stack: L-H-L
    for mat in ['L', 'H', 'L']:
        n = nL if mat == 'L' else nH
        d = wl_center / (4.0 * n)
        init_materials.append(mat)
        init_thicknesses.append(d)
    
    best_materials = list(init_materials)
    best_thicknesses = list(init_thicknesses)
    best_score = _penalized_score(best_materials, best_thicknesses, problem, wls, 
                                   materials_dict, n_amb, n_sub, band_nm)
    
    # Simulated annealing
    T_start = 5.0
    T_end = 0.01
    n_iterations = 3000  # Will be adjusted by time
    
    # Estimate time per iteration
    t0 = time.time()
    _sa_step(best_materials, best_thicknesses, problem, wls, materials_dict,
             n_amb, n_sub, band_nm, T_start, rng)
    t1 = time.time()
    time_per_iter = max(t1 - t0, 0.001)
    
    # Adjust iterations to fit within time budget
    time_budget = deadline - (time.time() - start_time) - 10.0  # Reserve 10s for refinement
    max_iter_sa = int(time_budget / time_per_iter)
    max_iter_sa = max(100, min(max_iter_sa, 10000))
    
    current_materials = list(init_materials)
    current_thicknesses = list(init_thicknesses)
    current_score = best_score
    
    for i in range(max_iter_sa):
        if time.time() - start_time > deadline - 15:
            break
        
        # Cooling schedule
        T = T_start * (T_end / T_start) ** (i / max_iter_sa)
        
        new_materials, new_thicknesses, accepted = _sa_step(
            current_materials, current_thicknesses, problem, wls, materials_dict,
            n_amb, n_sub, band_nm, T, rng)
        
        if accepted:
            current_materials = new_materials
            current_thicknesses = new_thicknesses
            current_score = _penalized_score(current_materials, current_thicknesses, problem, 
                                             wls, materials_dict, n_amb, n_sub, band_nm)
            
            if current_score < best_score:
                best_score = current_score
                best_materials = list(current_materials)
                best_thicknesses = list(current_thicknesses)
    
    # Final local refinement on the best solution
    if time.time() - start_time < deadline - 5 and len(best_materials) > 0:
        try:
            refined_materials, refined_thicknesses = _local_refine(
                list(best_materials), list(best_thicknesses), problem, wls, materials_dict,
                n_amb, n_sub, band_nm, max_iter=200)
            
            if _is_valid(refined_materials, refined_thicknesses, problem):
                refined_score = _mean_reflectance(refined_materials, refined_thicknesses, 
                                                   band_nm, 401, n_amb, n_sub, materials_dict)
                if refined_score < best_score:
                    best_score = refined_score
                    best_materials = refined_materials
                    best_thicknesses = refined_thicknesses
        except Exception:
            pass
    
    # Ensure validity
    if not _is_valid(best_materials, best_thicknesses, problem):
        # Fallback to a valid single layer
        best_materials = ["L"]
        best_thicknesses = [wl_center / (4.0 * nL)]
    
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
